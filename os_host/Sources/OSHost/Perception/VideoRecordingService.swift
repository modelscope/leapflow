import AppKit
import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit

/// Segment metadata returned after recording stops.
struct VideoSegmentInfo {
    let path: String
    let startTime: Double
    let durationS: Double
    let fps: Double
    let width: Int
    let height: Int
    let codec: String
    let fileSize: Int
}

/// Manages continuous screen recording sessions using ScreenCaptureKit + AVAssetWriter.
/// Supports pause/resume, automatic segment splitting by duration,
/// and mouse-driven dynamic display switching.
final class VideoRecordingService: NSObject, SCStreamOutput {

    // MARK: - State

    private var stream: SCStream?
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?

    private var outputDir: URL?
    private var currentSegmentURL: URL?
    private var segmentIndex: Int = 0
    private var segmentStartTime: CMTime = .zero
    private var firstSampleTime: CMTime = .zero
    private var recordingStartTime: CFAbsoluteTime = 0
    private var segments: [VideoSegmentInfo] = []

    private var configuredFps: Int = 5
    private var configuredResolutionScale: Double = 0.5
    private var configuredCodec: String = "h264"
    private var configuredMaxSegmentS: Int = 600
    private var captureWidth: Int = 0
    private var captureHeight: Int = 0

    private(set) var isRecording = false
    private(set) var isPaused = false
    private var frameCount: Int = 0
    private var segmentFrameCount: Int = 0
    private var writerStarted = false

    private let writingQueue = DispatchQueue(label: "com.leapflow.video.writing")

    // MARK: - Mouse-Driven Display Tracking

    private let dwellTracker: DisplayDwellTracker
    private var mousePollingTimer: DispatchSourceTimer?
    private var currentDisplayID: UInt32 = 0
    private var cachedDisplays: [SCDisplay] = []
    private let trackingConfig: MouseTrackingConfig
    private var isSwitchingDisplay = false

    // MARK: - Init

    init(config: MouseTrackingConfig = MouseTrackingConfig()) {
        self.trackingConfig = config
        self.dwellTracker = DisplayDwellTracker(config: config)
        super.init()
    }

    // MARK: - Public API

    func start(outputDir: URL, fps: Int, resolutionScale: Double, codec: String, maxSegmentS: Int) async throws {
        guard !isRecording else {
            throw VideoRecordingError.alreadyRecording
        }

        self.outputDir = outputDir
        self.configuredFps = fps
        self.configuredResolutionScale = resolutionScale
        self.configuredCodec = codec
        self.configuredMaxSegmentS = maxSegmentS
        self.segments = []
        self.segmentIndex = 0
        self.frameCount = 0
        self.isPaused = false

        // Ensure output directory exists
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        // Get available displays
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        cachedDisplays = content.displays

        // Determine initial display: prefer mouse position, fallback to frontmost app, then first
        let mousePos = NSEvent.mouseLocation
        let initialDisplayID = DisplayDwellTracker.displayIDForMousePosition(mousePos, displays: content.displays)
            ?? selectActiveDisplay(from: content).map { UInt32($0.displayID) }
            ?? content.displays.first.map { UInt32($0.displayID) }

        guard let displayID = initialDisplayID,
              let display = content.displays.first(where: { UInt32($0.displayID) == displayID }) else {
            throw VideoRecordingError.noDisplayAvailable
        }

        currentDisplayID = displayID
        dwellTracker.setInitialDisplay(displayID)

        // Configure capture dimensions
        captureWidth = Int(Double(display.width) * resolutionScale)
        captureHeight = Int(Double(display.height) * resolutionScale)
        // Ensure even dimensions for H.264 encoding
        captureWidth = captureWidth & ~1
        captureHeight = captureHeight & ~1

        // SCStream configuration
        let streamConfig = buildStreamConfiguration(for: display)

        // Content filter for full display
        let filter = SCContentFilter(display: display, excludingWindows: [])

        // Create stream
        let scStream = SCStream(filter: filter, configuration: streamConfig, delegate: nil)
        try scStream.addStreamOutput(self, type: .screen, sampleHandlerQueue: writingQueue)
        self.stream = scStream

        // Create first segment writer
        try createNewSegmentWriter()

        // Start capture
        try await scStream.startCapture()
        recordingStartTime = CFAbsoluteTimeGetCurrent()
        isRecording = true

        // Start mouse-driven display tracking
        startMousePolling()
    }

    func stop() async -> [VideoSegmentInfo] {
        guard isRecording else { return [] }
        isRecording = false
        isPaused = false
        isSwitchingDisplay = false

        // Stop mouse polling
        stopMousePolling()
        dwellTracker.reset()

        // Stop the stream
        if let stream = self.stream {
            try? await stream.stopCapture()
            self.stream = nil
        }

        // Finish current writer
        await finalizeCurrentWriter()

        let result = segments
        segments = []
        return result
    }

    func pause() {
        guard isRecording else { return }
        isPaused = true
    }

    func resume() {
        guard isRecording else { return }
        isPaused = false
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, isRecording, !isPaused else { return }

        guard sampleBuffer.isValid,
              CMSampleBufferGetNumSamples(sampleBuffer) > 0 else { return }

        // Check for status-only buffers (no image data)
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let statusRaw = attachments.first?[.status] as? Int,
              statusRaw == SCFrameStatus.complete.rawValue else {
            return
        }

        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        // Initialize timing on first frame
        if !writerStarted {
            firstSampleTime = presentationTime
            segmentStartTime = presentationTime
            assetWriter?.startSession(atSourceTime: presentationTime)
            writerStarted = true
        }

        // Check segment duration and rotate if needed
        let elapsed = CMTimeGetSeconds(CMTimeSubtract(presentationTime, segmentStartTime))
        if elapsed >= Double(configuredMaxSegmentS) {
            rotateSegment(at: presentationTime)
        }

        // Write the frame
        guard let input = videoInput, input.isReadyForMoreMediaData else { return }
        input.append(sampleBuffer)
        frameCount += 1
        segmentFrameCount += 1
    }

    // MARK: - Display Selection

    /// Select the display where the user is actively working (frontmost app fallback).
    private func selectActiveDisplay(from content: SCShareableContent) -> SCDisplay? {
        guard let frontApp = NSWorkspace.shared.frontmostApplication,
              let window = content.windows.first(where: {
                  $0.owningApplication?.processID == frontApp.processIdentifier && $0.isOnScreen
              }) else {
            return nil
        }
        return content.displays.first(where: { $0.frame.intersects(window.frame) })
    }

    // MARK: - Mouse Polling & Display Switching

    private func startMousePolling() {
        guard trackingConfig.displaySwitchEnabled else { return }

        let interval = 1.0 / trackingConfig.pollingFrequency
        let timer = DispatchSource.makeTimerSource(queue: writingQueue)
        timer.schedule(deadline: .now() + interval, repeating: interval)
        timer.setEventHandler { [weak self] in
            self?.sampleMousePosition()
        }
        timer.resume()
        mousePollingTimer = timer
    }

    private func stopMousePolling() {
        mousePollingTimer?.cancel()
        mousePollingTimer = nil
    }

    private func sampleMousePosition() {
        let mousePos = NSEvent.mouseLocation

        if let displayID = DisplayDwellTracker.displayIDForMousePosition(mousePos, displays: cachedDisplays) {
            dwellTracker.recordSample(displayID: displayID)

            // Check if we should switch displays
            if let newDisplayID = dwellTracker.detectActiveDisplay() {
                handleDisplaySwitch(to: newDisplayID)
            }
        }
    }

    private func handleDisplaySwitch(to newDisplayID: UInt32) {
        guard isRecording, !isPaused, newDisplayID != currentDisplayID,
              !isSwitchingDisplay else { return }

        fputs("OSHost: VideoRecording switching to display \(newDisplayID)\n", stderr)

        // Guard against overlapping switches
        isSwitchingDisplay = true

        // Perform segment cut + stream switch
        Task {
            await switchStreamToDisplay(newDisplayID)
            self.isSwitchingDisplay = false
        }
    }

    private func switchStreamToDisplay(_ displayID: UInt32) async {
        // Pause writing to prevent delegate from accessing writer during transition
        isPaused = true

        // 1. Finalize current segment
        finalizeCurrentWriterSync()

        // 2. Stop current stream
        if let oldStream = stream {
            try? await oldStream.stopCapture()
            stream = nil
        }

        // 3. Refresh display list and find target display
        guard let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true),
              let newDisplay = content.displays.first(where: { UInt32($0.displayID) == displayID }) else {
            fputs("OSHost: Display \(displayID) not found for switch\n", stderr)
            return
        }
        cachedDisplays = content.displays

        // 4. Reconfigure capture dimensions for new display
        captureWidth = Int(Double(newDisplay.width) * configuredResolutionScale)
        captureHeight = Int(Double(newDisplay.height) * configuredResolutionScale)
        captureWidth = captureWidth & ~1
        captureHeight = captureHeight & ~1

        // 5. Create new stream
        let streamConfig = buildStreamConfiguration(for: newDisplay)
        let filter = SCContentFilter(display: newDisplay, excludingWindows: [])
        let newStream = SCStream(filter: filter, configuration: streamConfig, delegate: nil)

        do {
            try newStream.addStreamOutput(self, type: .screen, sampleHandlerQueue: writingQueue)
            try await newStream.startCapture()
        } catch {
            fputs("OSHost: Failed to start capture on display \(displayID): \(error)\n", stderr)
            return
        }

        stream = newStream
        currentDisplayID = displayID

        // 6. Start new segment
        segmentIndex += 1
        do {
            try createNewSegmentWriter()
        } catch {
            fputs("OSHost: Failed to create segment after display switch: \(error)\n", stderr)
        }

        // Resume writing now that new stream + writer are ready
        isPaused = false
    }

    // MARK: - Stream Configuration

    private func buildStreamConfiguration(for display: SCDisplay) -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        var width = Int(Double(display.width) * configuredResolutionScale)
        var height = Int(Double(display.height) * configuredResolutionScale)
        // Ensure even dimensions for H.264 encoding
        if width % 2 != 0 { width += 1 }
        if height % 2 != 0 { height += 1 }
        config.width = width
        config.height = height
        config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(configuredFps))
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = true
        return config
    }

    // MARK: - Segment Management

    private func createNewSegmentWriter() throws {
        let fileName = String(format: "segment_%03d.mp4", segmentIndex)
        guard let dir = outputDir else {
            throw VideoRecordingError.noOutputDirectory
        }
        let segmentURL = dir.appendingPathComponent(fileName)
        currentSegmentURL = segmentURL

        // Remove existing file if present
        try? FileManager.default.removeItem(at: segmentURL)

        let writer = try AVAssetWriter(url: segmentURL, fileType: .mp4)

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: captureWidth,
            AVVideoHeightKey: captureHeight,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: captureWidth * captureHeight * 2,
                AVVideoExpectedSourceFrameRateKey: configuredFps,
            ] as [String: Any],
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = true

        let sourcePixelAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: captureWidth,
            kCVPixelBufferHeightKey as String: captureHeight,
        ]
        let pixelAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: sourcePixelAttrs
        )

        writer.add(input)
        writer.startWriting()
        // startSession will be called on first frame arrival

        self.assetWriter = writer
        self.videoInput = input
        self.adaptor = pixelAdaptor
        self.writerStarted = false
        self.segmentFrameCount = 0
    }

    private func rotateSegment(at time: CMTime) {
        // Finalize current segment synchronously on the writing queue (already on it)
        finalizeCurrentWriterSync()

        // Start new segment
        segmentIndex += 1
        segmentStartTime = time
        do {
            try createNewSegmentWriter()
            // Start session at the rotation time
            assetWriter?.startSession(atSourceTime: time)
            writerStarted = true
        } catch {
            fputs("OSHost: VideoRecordingService failed to create new segment: \(error)\n", stderr)
        }
    }

    private func finalizeCurrentWriterSync() {
        guard let writer = assetWriter, let url = currentSegmentURL else { return }
        videoInput?.markAsFinished()

        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting {
            semaphore.signal()
        }
        semaphore.wait()

        // Collect segment metadata
        let duration: Double
        if segmentFrameCount > 0 {
            duration = Double(segmentFrameCount) / Double(configuredFps)
        } else {
            duration = 0
        }
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0

        // Use cumulative duration of previous segments as start offset
        // (consistent with the async finalizeCurrentWriter method)
        let segStartOffset: Double
        if segments.isEmpty {
            segStartOffset = 0
        } else {
            segStartOffset = segments.reduce(0) { $0 + $1.durationS }
        }

        let info = VideoSegmentInfo(
            path: url.path,
            startTime: segStartOffset,
            durationS: duration,
            fps: Double(configuredFps),
            width: captureWidth,
            height: captureHeight,
            codec: configuredCodec,
            fileSize: fileSize
        )
        segments.append(info)

        self.assetWriter = nil
        self.videoInput = nil
        self.adaptor = nil
    }

    private func finalizeCurrentWriter() async {
        guard let writer = assetWriter, let url = currentSegmentURL else { return }
        videoInput?.markAsFinished()

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting {
                continuation.resume()
            }
        }

        // Collect segment metadata
        let duration: Double
        if segmentFrameCount > 0 {
            duration = Double(segmentFrameCount) / Double(configuredFps)
        } else {
            duration = 0
        }
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0

        let segStartOffset: Double
        if segments.isEmpty {
            segStartOffset = 0
        } else {
            segStartOffset = segments.reduce(0) { $0 + $1.durationS }
        }

        let info = VideoSegmentInfo(
            path: url.path,
            startTime: segStartOffset,
            durationS: duration,
            fps: Double(configuredFps),
            width: captureWidth,
            height: captureHeight,
            codec: configuredCodec,
            fileSize: fileSize
        )
        segments.append(info)

        self.assetWriter = nil
        self.videoInput = nil
        self.adaptor = nil
    }
}

// MARK: - Errors

enum VideoRecordingError: Error, CustomStringConvertible {
    case alreadyRecording
    case noDisplayAvailable
    case noOutputDirectory
    case permissionDenied

    var description: String {
        switch self {
        case .alreadyRecording:
            return "Recording is already in progress"
        case .noDisplayAvailable:
            return "No display available for recording"
        case .noOutputDirectory:
            return "Output directory not configured"
        case .permissionDenied:
            return "Screen recording permission denied"
        }
    }
}

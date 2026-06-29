import AppKit
import CoreGraphics
import Foundation
import ScreenCaptureKit

// MARK: - Data Types

struct CaptureConfig {
    var quality: CGFloat = 0.75
    var scaleFactor: CGFloat = 1.0
    var format: CaptureFormat = .jpeg
}

enum CaptureFormat {
    case jpeg
    case png
}

struct FrameData {
    let data: Data
    let base64: String
    let width: Int
    let height: Int
    let timestamp: TimeInterval
}

enum CaptureError: Error, CustomStringConvertible {
    case permissionDenied
    case captureFailure(String)
    case encodingFailure
    case noDisplayAvailable

    var description: String {
        switch self {
        case .permissionDenied:
            return "Screen capture permission denied"
        case .captureFailure(let reason):
            return "Capture failed: \(reason)"
        case .encodingFailure:
            return "Image encoding failed"
        case .noDisplayAvailable:
            return "No display available for capture"
        }
    }
}

enum PermissionState {
    case granted
    case denied
    case notDetermined
}

// MARK: - ScreenCaptureMonitor

/// Provides screen capture capabilities using macOS ScreenCaptureKit.
/// Captures full-screen or region frames, encodes to JPEG/PNG base64,
/// and optionally broadcasts capture events to connected clients.
final class ScreenCaptureMonitor {

    // MARK: - Dependencies

    private let broadcaster: ClientBroadcaster

    // MARK: - Configuration

    private let config: CaptureConfig

    // MARK: - State

    private let lock = NSLock()
    private var _isCapturing = false
    private var captureStream: SCStream?
    private var streamDelegate: StreamOutputHandler?

    var isCapturing: Bool {
        lock.withLock { _isCapturing }
    }

    private func setCapturing(stream: SCStream, delegate: StreamOutputHandler) {
        lock.withLock {
            self.captureStream = stream
            self.streamDelegate = delegate
            self._isCapturing = true
        }
    }

    private func clearCapturing() -> SCStream? {
        lock.withLock {
            guard _isCapturing else { return nil }
            _isCapturing = false
            let stream = captureStream
            captureStream = nil
            streamDelegate = nil
            return stream
        }
    }

    private func tryBeginCapture() -> Bool {
        lock.withLock {
            guard !_isCapturing else { return false }
            return true
        }
    }

    // MARK: - Init

    init(broadcaster: ClientBroadcaster, config: CaptureConfig = CaptureConfig()) {
        self.broadcaster = broadcaster
        self.config = config
    }

    deinit {
        stopCapture()
    }

    // MARK: - Permission

    func permissionStatus() -> PermissionState {
        if CGPreflightScreenCaptureAccess() {
            return .granted
        }
        return .notDetermined
    }

    func requestPermission() async -> Bool {
        if #available(macOS 15.0, *) {
            if CGPreflightScreenCaptureAccess() {
                return true
            }
            CGRequestScreenCaptureAccess()
            // The system shows a prompt; return current state after request
            return CGPreflightScreenCaptureAccess()
        } else {
            // On macOS 13/14, trigger permission dialog via SCShareableContent query
            do {
                _ = try await SCShareableContent.excludingDesktopWindows(
                    false, onScreenWindowsOnly: true)
                return true
            } catch {
                fputs("ScreenCaptureMonitor: permission request failed: \(error)\n", stderr)
                return false
            }
        }
    }

    // MARK: - Capture

    /// Unified capture entry point. If `region` is nil, captures full screen.
    func captureFrame(region: CGRect? = nil) async throws -> FrameData {
        if let region = region {
            return try await captureRegion(region)
        }
        return try await captureFullScreen()
    }

    /// Captures the display containing the frontmost app (falls back to primary).
    func captureFullScreen(content existingContent: SCShareableContent? = nil) async throws -> FrameData {
        let content: SCShareableContent
        if let existing = existingContent {
            content = existing
        } else {
            content = try await fetchShareableContent()
        }

        guard let display = displayForFrontmostApp(content: content)
                ?? content.displays.first else {
            throw CaptureError.noDisplayAvailable
        }

        let cgImage = try await captureDisplay(display, content: content)
        let frameData = try encode(cgImage: cgImage)

        broadcastCaptureEvent(frameData)
        return frameData
    }

    /// Captures a specific app window by bundle ID (display-independent).
    func captureWindow(bundleId: String) async throws -> FrameData {
        let content = try await fetchShareableContent()

        guard let window = content.windows.first(where: {
            $0.owningApplication?.bundleIdentifier == bundleId && $0.isOnScreen
        }) else {
            return try await captureFullScreen(content: content)
        }

        let cgImage = try await captureWindowImage(window, content: content)
        let frameData = try encode(cgImage: cgImage)
        broadcastCaptureEvent(frameData)
        return frameData
    }

    /// Captures a specific region of the display containing the frontmost app.
    func captureRegion(_ region: CGRect) async throws -> FrameData {
        let content = try await fetchShareableContent()

        guard let display = displayForFrontmostApp(content: content)
                ?? content.displays.first else {
            throw CaptureError.noDisplayAvailable
        }

        let fullImage = try await captureDisplay(display, content: content)

        // Convert region to pixel coordinates accounting for scale factor
        let scale = CGFloat(fullImage.width) / CGFloat(display.width)
        let scaledRect = CGRect(
            x: region.origin.x * scale,
            y: region.origin.y * scale,
            width: region.size.width * scale,
            height: region.size.height * scale
        )

        guard let cropped = fullImage.cropping(to: scaledRect) else {
            throw CaptureError.captureFailure("Failed to crop image to region \(region)")
        }

        let frameData = try encode(cgImage: cropped)
        broadcastCaptureEvent(frameData)
        return frameData
    }

    // MARK: - Continuous Capture

    /// Starts continuous capture mode using SCStream.
    func startCapture() async throws {
        guard tryBeginCapture() else { return }

        let content = try await fetchShareableContent()

        guard let display = displayForFrontmostApp(content: content)
                ?? content.displays.first else {
            throw CaptureError.noDisplayAvailable
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let streamConfig = SCStreamConfiguration()
        streamConfig.width = Int(CGFloat(display.width) * config.scaleFactor)
        streamConfig.height = Int(CGFloat(display.height) * config.scaleFactor)
        streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: 2) // 2 FPS max
        streamConfig.queueDepth = 3

        let handler = StreamOutputHandler(monitor: self)
        let stream = SCStream(filter: filter, configuration: streamConfig, delegate: nil)
        try stream.addStreamOutput(handler, type: .screen, sampleHandlerQueue: .global(qos: .utility))
        try await stream.startCapture()

        setCapturing(stream: stream, delegate: handler)

        fputs("ScreenCaptureMonitor: continuous capture started\n", stderr)
    }

    /// Stops continuous capture mode.
    func stopCapture() {
        guard let stream = clearCapturing() else { return }

        Task {
            do {
                try await stream.stopCapture()
            } catch {
                fputs("ScreenCaptureMonitor: error stopping capture: \(error)\n", stderr)
            }
        }

        fputs("ScreenCaptureMonitor: capture stopped\n", stderr)
    }

    // MARK: - Private Helpers

    private func fetchShareableContent() async throws -> SCShareableContent {
        do {
            return try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
        } catch {
            if (error as NSError).code == -3801 {
                throw CaptureError.permissionDenied
            }
            throw CaptureError.captureFailure("Failed to get shareable content: \(error)")
        }
    }

    private func displayForFrontmostApp(content: SCShareableContent) -> SCDisplay? {
        guard let frontApp = NSWorkspace.shared.frontmostApplication,
              let window = content.windows.first(where: {
                  $0.owningApplication?.processID == frontApp.processIdentifier && $0.isOnScreen
              }) else { return nil }
        return content.displays.first(where: { $0.frame.intersects(window.frame) })
    }

    private func captureWindowImage(_ window: SCWindow, content: SCShareableContent) async throws -> CGImage {
        if #available(macOS 14.0, *) {
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let streamConfig = SCStreamConfiguration()
            streamConfig.width = Int(window.frame.width * config.scaleFactor)
            streamConfig.height = Int(window.frame.height * config.scaleFactor)
            return try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: streamConfig)
        } else {
            // macOS 13 fallback: capture the display containing the window, then crop
            guard let display = content.displays.first(where: {
                $0.frame.intersects(window.frame)
            }) ?? content.displays.first else {
                throw CaptureError.noDisplayAvailable
            }
            let fullImage = try await captureViaSingleFrameStream(display: display)
            let scale = CGFloat(fullImage.width) / CGFloat(display.width)
            // CGImage.cropping uses top-left origin; SCWindow.frame uses Cocoa bottom-left
            let windowLocalY = window.frame.origin.y - display.frame.origin.y
            let localFrame = CGRect(
                x: (window.frame.origin.x - display.frame.origin.x) * scale,
                y: (CGFloat(display.height) - windowLocalY - window.frame.height) * scale,
                width: window.frame.width * scale,
                height: window.frame.height * scale
            )
            guard let cropped = fullImage.cropping(to: localFrame) else {
                return fullImage
            }
            return cropped
        }
    }

    private func captureDisplay(_ display: SCDisplay, content: SCShareableContent) async throws -> CGImage {
        if #available(macOS 14.0, *) {
            return try await captureViaScreenshotManager(display: display, content: content)
        } else {
            return try await captureViaSingleFrameStream(display: display)
        }
    }

    @available(macOS 14.0, *)
    private func captureViaScreenshotManager(
        display: SCDisplay, content: SCShareableContent
    ) async throws -> CGImage {
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let streamConfig = SCStreamConfiguration()
        streamConfig.width = Int(CGFloat(display.width) * config.scaleFactor)
        streamConfig.height = Int(CGFloat(display.height) * config.scaleFactor)

        do {
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: streamConfig)
            return image
        } catch {
            throw CaptureError.captureFailure("SCScreenshotManager failed: \(error)")
        }
    }

    /// Fallback for macOS 13: use SCStream to capture a single frame.
    private func captureViaSingleFrameStream(display: SCDisplay) async throws -> CGImage {
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let streamConfig = SCStreamConfiguration()
        streamConfig.width = Int(CGFloat(display.width) * config.scaleFactor)
        streamConfig.height = Int(CGFloat(display.height) * config.scaleFactor)
        streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: 30)
        streamConfig.queueDepth = 1

        return try await withCheckedThrowingContinuation { continuation in
            let handler = SingleFrameHandler(continuation: continuation)
            let stream = SCStream(filter: filter, configuration: streamConfig, delegate: nil)

            do {
                try stream.addStreamOutput(handler, type: .screen,
                                           sampleHandlerQueue: .global(qos: .userInitiated))
            } catch {
                continuation.resume(throwing: CaptureError.captureFailure(
                    "Failed to add stream output: \(error)"))
                return
            }

            Task {
                do {
                    try await stream.startCapture()
                    // Give time for one frame, then stop
                    try await Task.sleep(nanoseconds: 500_000_000)
                    try await stream.stopCapture()
                    if !handler.didResume {
                        continuation.resume(throwing: CaptureError.captureFailure(
                            "No frame received from stream"))
                    }
                } catch {
                    if !handler.didResume {
                        continuation.resume(throwing: CaptureError.captureFailure(
                            "Stream capture failed: \(error)"))
                    }
                }
            }
        }
    }

    private func encode(cgImage: CGImage) throws -> FrameData {
        let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
        let imageData: Data?

        switch config.format {
        case .jpeg:
            imageData = bitmapRep.representation(
                using: .jpeg,
                properties: [.compressionFactor: config.quality])
        case .png:
            imageData = bitmapRep.representation(using: .png, properties: [:])
        }

        guard let data = imageData else {
            throw CaptureError.encodingFailure
        }

        let base64 = data.base64EncodedString()

        return FrameData(
            data: data,
            base64: base64,
            width: cgImage.width,
            height: cgImage.height,
            timestamp: Date().timeIntervalSince1970
        )
    }

    // MARK: - Event Broadcasting

    private func broadcastCaptureEvent(_ frame: FrameData) {
        let payload: [String: MPValue] = [
            "width": .int(Int64(frame.width)),
            "height": .int(Int64(frame.height)),
            "format": .string(config.format == .jpeg ? "jpeg" : "png"),
            "size_bytes": .int(Int64(frame.data.count)),
            "timestamp": .double(frame.timestamp),
        ]

        let eventFrame: [String: MPValue] = [
            "v": .int(1),
            "type": .string("event"),
            "event": .string("event.screen_frame_captured"),
            "payload": .map(payload),
        ]

        guard let data = try? MessageCodec.encodeFrame(eventFrame) else { return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.broadcaster.broadcast(data: data)
        }
    }

    // MARK: - Stream Handlers

    /// Handles continuous capture stream output.
    private final class StreamOutputHandler: NSObject, SCStreamOutput {
        private weak var monitor: ScreenCaptureMonitor?

        init(monitor: ScreenCaptureMonitor) {
            self.monitor = monitor
        }

        func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                    of type: SCStreamOutputType) {
            guard type == .screen else { return }
            guard let monitor = monitor else { return }
            guard let imageBuffer = sampleBuffer.imageBuffer else { return }

            let ciImage = CIImage(cvPixelBuffer: imageBuffer)
            let context = CIContext()
            let rect = CGRect(x: 0, y: 0,
                              width: CVPixelBufferGetWidth(imageBuffer),
                              height: CVPixelBufferGetHeight(imageBuffer))

            guard let cgImage = context.createCGImage(ciImage, from: rect) else { return }

            if let frameData = try? monitor.encode(cgImage: cgImage) {
                monitor.broadcastCaptureEvent(frameData)
            }
        }
    }

    /// Captures a single frame from an SCStream then resumes a continuation.
    private final class SingleFrameHandler: NSObject, SCStreamOutput {
        private let continuation: CheckedContinuation<CGImage, Error>
        private(set) var didResume = false

        init(continuation: CheckedContinuation<CGImage, Error>) {
            self.continuation = continuation
        }

        func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                    of type: SCStreamOutputType) {
            guard type == .screen, !didResume else { return }
            guard let imageBuffer = sampleBuffer.imageBuffer else { return }

            let ciImage = CIImage(cvPixelBuffer: imageBuffer)
            let context = CIContext()
            let rect = CGRect(x: 0, y: 0,
                              width: CVPixelBufferGetWidth(imageBuffer),
                              height: CVPixelBufferGetHeight(imageBuffer))

            guard let cgImage = context.createCGImage(ciImage, from: rect) else {
                didResume = true
                continuation.resume(throwing: CaptureError.captureFailure(
                    "Failed to create CGImage from pixel buffer"))
                return
            }

            didResume = true
            continuation.resume(returning: cgImage)
        }
    }
}

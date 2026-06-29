import AppKit
import AVFoundation
import CoreServices
import Foundation

/// Decodes `FSEventStreamEventFlags` bitmasks into stable semantic action names.
///
/// Table-driven: extending coverage requires only appending a `(flag, name)`
/// row, no branching changes. The decoder is independent of any RPC concern
/// so it can be reused by other watchers/tests (SRP).
struct FSEventFlagDecoder {
    /// Ordered list of `(flagBit, semanticName)`.
    /// Order is preserved in the decoded output to give consumers a stable
    /// iteration sequence (e.g. "created" before "modified").
    private static let flagTable: [(flag: UInt32, name: String)] = [
        (UInt32(kFSEventStreamEventFlagItemCreated),       "created"),
        (UInt32(kFSEventStreamEventFlagItemRemoved),       "deleted"),
        (UInt32(kFSEventStreamEventFlagItemRenamed),       "renamed"),
        (UInt32(kFSEventStreamEventFlagItemModified),      "modified"),
        (UInt32(kFSEventStreamEventFlagItemInodeMetaMod),  "inode_meta"),
        (UInt32(kFSEventStreamEventFlagItemFinderInfoMod), "finder_info"),
        (UInt32(kFSEventStreamEventFlagItemXattrMod),      "xattr_mod"),
        (UInt32(kFSEventStreamEventFlagMustScanSubDirs),   "must_scan"),
        (UInt32(kFSEventStreamEventFlagUserDropped),       "user_dropped"),
        (UInt32(kFSEventStreamEventFlagKernelDropped),     "kernel_dropped"),
        (UInt32(kFSEventStreamEventFlagRootChanged),       "root_changed"),
        (UInt32(kFSEventStreamEventFlagMount),             "mount"),
        (UInt32(kFSEventStreamEventFlagUnmount),           "unmount"),
    ]

    /// Returns the set of semantic action names whose flag bits are set in `flags`.
    /// Unknown bits are intentionally ignored — the caller still has the raw
    /// `flags` value available for forensic purposes.
    static func decode(_ flags: UInt32) -> [String] {
        flagTable.compactMap { entry in (flags & entry.flag) != 0 ? entry.name : nil }
    }
}

/// Domain-based RPC router. Each method is dispatched to the responsible provider.
/// Replaces the monolithic switch with a two-level domain/action split (SRP).
final class RpcRouter {
    internal var capabilities: PlatformCapabilities
    private let uiSensor: UISensorProvider
    private var fileWatcher: FileWatchProvider
    private let fileOps: FileOperationProvider
    private let clipboard: ClipboardProvider
    private let appControl: AppControlProvider
    private let intentProvider: IntentProvider
    private let shell: ShellProvider
    private let input: InputProvider
    internal var screenCaptureProvider: ScreenCaptureProvider?
    var videoService = VideoRecordingService()
    let audit = AuditLog()
    var broadcaster: ClientBroadcaster?
    var appFocusMonitor: AppFocusMonitor?
    let recordingMode = RecordingMode()

    private let lock = NSLock()
    private var fsLog: [(path: String, flags: UInt32, ts: TimeInterval)] = []

    init() {
        self.capabilities = CapabilityDetector.detect()
        self.uiSensor = ProviderFactory.makeUISensor()
        self.fileOps = ProviderFactory.makeFileOperator()
        self.clipboard = ProviderFactory.makeClipboard()
        self.appControl = ProviderFactory.makeAppControl()
        self.intentProvider = ProviderFactory.makeIntentProvider()
        self.shell = ProviderFactory.makeShellProvider()
        self.input = ProviderFactory.makeInputProvider()

        self.screenCaptureProvider = nil  // Injected from main.swift after init

        self.fileWatcher = ProviderFactory.makeFileWatcher { _, _, _ in }
        self.fileWatcher = ProviderFactory.makeFileWatcher { [weak self] path, flags, ts in
            self?.recordFsEvent(path: path, flags: flags, ts: ts)
        }
        _ = fileWatcher.subscribe(path: NSHomeDirectory())
    }

    private func recordFsEvent(path: String, flags: UInt32, ts: TimeInterval) {
        let coalesceWindow = recordingMode.fsCoalesceWindow
        lock.lock()
        if let last = fsLog.last, last.path == path,
           ts - last.ts < coalesceWindow {
            fsLog[fsLog.count - 1] = (path, last.flags | flags, ts)
            lock.unlock()
            return
        }
        fsLog.append((path, flags, ts))
        if fsLog.count > 500 {
            fsLog.removeFirst(fsLog.count - 500)
        }
        lock.unlock()
        let semanticActions = FSEventFlagDecoder.decode(flags)
        pushEvent(type: "event.fs_change", payload: [
            "path": .string(path),
            "flags": .uint(UInt64(flags)),
            "semantic_actions": .array(semanticActions.map { .string($0) }),
            "ts": .double(ts),
        ])
    }

    // MARK: - Event Push

    private func pushEvent(type: String, payload: [String: MPValue]) {
        guard let broadcaster else { return }
        var finalPayload = payload
        if recordingMode.isActive {
            finalPayload["_seq"] = .uint(recordingMode.nextSequence())
        }
        let frame: [String: MPValue] = [
            "v": .int(1),
            "type": .string("event"),
            "event": .string(type),
            "payload": .map(finalPayload),
            "ts": .double(ProcessInfo.processInfo.systemUptime),
        ]
        guard let data = try? MessageCodec.encodeFrame(frame) else { return }
        DispatchQueue.global(qos: .utility).async {
            broadcaster.broadcast(data: data)
        }
    }

    /// Start polling clipboard for changes and pushing events.
    func startClipboardPolling() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            var lastCount = NSPasteboard.general.changeCount
            while true {
                Thread.sleep(forTimeInterval: 1.0)
                guard let self else { return }
                let current = NSPasteboard.general.changeCount
                if current != lastCount {
                    lastCount = current
                    let text = NSPasteboard.general.string(forType: .string) ?? ""
                    self.pushEvent(type: "event.clipboard_change", payload: [
                        "text": .string(text),
                        "change_count": .int(Int64(current)),
                        "change_ts": .double(Date().timeIntervalSince1970),
                    ])
                }
            }
        }
    }

    // MARK: - RPC Frame Handler

    func handleRpc(frame: [String: MPValue]) throws -> Data {
        let rid = frame["id"]?.asString() ?? ""
        let method = frame["method"]?.asString() ?? ""
        let params = frame["params"]?.asMap() ?? [:]

        audit.record(method: method, detail: ["id": rid])

        let result: MPValue
        do {
            result = try route(method: method, params: params)
            return try MessageCodec.encodeFrame([
                "v": .int(1),
                "type": .string("response"),
                "id": .string(rid),
                "ok": .bool(true),
                "result": result,
            ])
        } catch let e as HostRpcError {
            fputs("OSHost: RPC error \(method): [\(e.code)] \(e.message)\n", stderr)
            return try MessageCodec.encodeFrame([
                "v": .int(1),
                "type": .string("response"),
                "id": .string(rid),
                "ok": .bool(false),
                "error": .map([
                    "code": .string(e.code),
                    "message": .string(e.message),
                    "details": .map(e.details),
                ]),
            ])
        }
    }

    // MARK: - Domain Router

    private func route(method: String, params: [String: MPValue]) throws -> MPValue {
        switch method {
        case "ping":
            return .map(["pong": .bool(true)])

        case "system.info":
            return handleSystemInfo()
        case "system.manifest":
            return capabilities.toMPValue()

        case "file.list":
            return try handleFileList(params)
        case "file.move":
            return try handleFileMove(params)
        case "file.copy":
            return try handleFileCopy(params)
        case "file.delete":
            return try handleFileDelete(params)

        case "fs.subscribe":
            return handleFsSubscribe(params)

        case "ax.tree":
            return handleAXTree(params)
        case "ax.perform":
            return handleAXPerform(params)

        case "app.launch":
            return try handleAppLaunch(params)
        case "app.activate":
            return try handleAppActivate(params)
        case "app.list":
            return try handleAppList(params)

        case "clipboard.get":
            return handleClipboardGet()
        case "clipboard.set":
            return handleClipboardSet(params)
        case "clipboard.last_change":
            return handleClipboardLastChange()

        case "input.type_text":
            return handleInputTypeText(params)
        case "input.shortcut":
            return handleInputShortcut(params)

        case "intent.discover":
            return handleIntentDiscover(params)
        case "intent.perform":
            return handleIntentPerform(params)

        case "screen.start_capture":
            return try handleScreenStartCapture(params)
        case "screen.stop_capture":
            return try handleScreenStopCapture()
        case "screen.capture_frame":
            return try handleScreenCaptureFrame(params)
        case "screen.permission_status":
            return handleScreenPermissionStatus()

        case "recording.start":
            return handleRecordingStart(params)
        case "recording.stop":
            return handleRecordingStop()

        case "video.start":
            return try handleVideoStart(params)
        case "video.stop":
            return try handleVideoStop()
        case "video.pause":
            return try handleVideoPause()
        case "video.resume":
            return try handleVideoResume()
        case "video.extract_frame":
            return try handleVideoExtractFrame(params)

        default:
            throw HostRpcError(code: "unsupported_method", message: method, details: [:])
        }
    }

    // MARK: - System

    private func handleSystemInfo() -> MPValue {
        .map([
            "platform": .string("darwin"),
            "platform_id": .string(capabilities.platformId),
            "os_version": .string(capabilities.osVersion),
            "accessibility_trusted": .bool(
                PermissionGuard.checkAccessibilityTrusted(prompt: false)
            ),
            "hostname": .string(ProcessInfo.processInfo.hostName),
        ])
    }

    // MARK: - File Operations

    private func handleFileList(_ params: [String: MPValue]) throws -> MPValue {
        let path = params["path"]?.asString() ?? NSHomeDirectory()
        let hidden = params["include_hidden"]?.asBool() ?? false
        fputs("OSHost: file.list path=\(path)\n", stderr)
        do {
            let result = try fileOps.listDirectory(path: path, includeHidden: hidden)
            if case .map(let m) = result, case .array(let arr)? = m["entries"] {
                fputs("OSHost: file.list → \(arr.count) entries\n", stderr)
            }
            return result
        } catch {
            fputs("OSHost: file.list FAILED: \(error.localizedDescription)\n", stderr)
            throw HostRpcError(code: "file_list_failed", message: error.localizedDescription, details: [:])
        }
    }

    private func handleFileMove(_ params: [String: MPValue]) throws -> MPValue {
        let src = params["src"]?.asString() ?? ""
        let dst = params["dst"]?.asString() ?? ""
        guard !src.isEmpty, !dst.isEmpty else {
            throw HostRpcError(code: "invalid_params", message: "src/dst required", details: [:])
        }
        fputs("OSHost: file.move \(src) → \(dst)\n", stderr)
        do {
            let result = try fileOps.moveItem(src: src, dst: dst)
            audit.record(method: "file.move", detail: ["src": src, "dst": dst])
            return result
        } catch {
            fputs("OSHost: file.move FAILED: \(error.localizedDescription)\n", stderr)
            throw HostRpcError(code: "file_move_failed", message: error.localizedDescription, details: [:])
        }
    }

    private func handleFileCopy(_ params: [String: MPValue]) throws -> MPValue {
        let src = params["src"]?.asString() ?? ""
        let dst = params["dst"]?.asString() ?? ""
        guard !src.isEmpty, !dst.isEmpty else {
            throw HostRpcError(code: "invalid_params", message: "src/dst required", details: [:])
        }
        do {
            return try fileOps.copyItem(src: src, dst: dst)
        } catch {
            throw HostRpcError(code: "file_copy_failed", message: error.localizedDescription, details: [:])
        }
    }

    private func handleFileDelete(_ params: [String: MPValue]) throws -> MPValue {
        let path = params["path"]?.asString() ?? ""
        guard !path.isEmpty else {
            throw HostRpcError(code: "invalid_params", message: "path required", details: [:])
        }
        do {
            return try fileOps.deleteItem(path: path)
        } catch {
            throw HostRpcError(code: "file_delete_failed", message: error.localizedDescription, details: [:])
        }
    }

    // MARK: - File System Watch

    private func handleFsSubscribe(_ params: [String: MPValue]) -> MPValue {
        let path = params["path"]?.asString() ?? NSHomeDirectory()
        let sid = fileWatcher.subscribe(path: path)
        audit.record(method: "fs.subscribe", detail: ["path": path])
        return .map([
            "subscription_id": .string(sid),
            "path": .string((path as NSString).expandingTildeInPath),
            "recent": .array(fileWatcher.recentEvents(limit: 20)),
        ])
    }

    // MARK: - UI / Accessibility

    private func handleAXTree(_ params: [String: MPValue]) -> MPValue {
        let bundle = params["bundle_id"]?.asString()
        let preferIntents = params["prefer_intents"]?.asBool() ?? false
        return uiSensor.readTree(bundleId: bundle, preferIntents: preferIntents)
    }

    private func handleAXPerform(_ params: [String: MPValue]) -> MPValue {
        let bundle = params["bundle_id"]?.asString() ?? ""
        let nodeId = params["node_id"]?.asString() ?? ""
        let action = params["action"]?.asString() ?? ""

        // Shell command support (legacy compatibility)
        if case .array(let cmds) = params["commands"] {
            var output = ""
            var exitCode: Int32 = 0
            for item in cmds {
                guard case .map(let m) = item,
                      case .string(let typ) = m["type"],
                      typ == "shell",
                      case .string(let cmd) = m["cmd"]
                else { continue }
                let result = (try? shell.executeWithStatus(command: cmd)) ?? ShellResult(output: "", exitCode: 1)
                output = result.output
                exitCode = result.exitCode
                audit.record(method: "ax.perform.shell", detail: ["bundle_id": bundle, "cmd": cmd])
            }
            return .map([
                "ok": .bool(exitCode == 0),
                "exit_code": .int(Int64(exitCode)),
                "bundle_id": .string(bundle),
                "stdout": .string(output),
            ])
        }

        // UI action via provider
        if !nodeId.isEmpty || !action.isEmpty {
            return uiSensor.performAction(nodeId: nodeId, action: action, params: params)
        }

        return .map(["ok": .bool(true), "bundle_id": .string(bundle)])
    }

    // MARK: - App Control

    private func handleAppLaunch(_ params: [String: MPValue]) throws -> MPValue {
        let bid = params["bundle_id"]?.asString() ?? ""
        guard !bid.isEmpty else {
            throw HostRpcError(code: "invalid_params", message: "bundle_id required", details: [:])
        }
        let ok = appControl.launch(bundleId: bid)
        return .map(["ok": .bool(ok), "bundle_id": .string(bid)])
    }

    private func handleAppActivate(_ params: [String: MPValue]) throws -> MPValue {
        let bid = params["bundle_id"]?.asString() ?? ""
        guard !bid.isEmpty else {
            throw HostRpcError(code: "invalid_params", message: "bundle_id required", details: [:])
        }
        let ok = appControl.activate(bundleId: bid)
        return .map(["ok": .bool(ok), "bundle_id": .string(bid)])
    }

    private func handleAppList(_ params: [String: MPValue]) throws -> MPValue {
        let filter = params["filter"]?.asString() ?? ""
        let runningOnly: Bool
        if case .bool(let v) = params["running_only"] {
            runningOnly = v
        } else {
            runningOnly = false
        }
        let apps = appControl.listApps(filter: filter, runningOnly: runningOnly)
        return .map(["ok": .bool(true), "apps": .array(apps)])
    }

    // MARK: - Clipboard

    private func handleClipboardGet() -> MPValue {
        let (text, count, ts) = clipboard.snapshot()
        return .map([
            "text": .string(text),
            "change_count": .int(Int64(count)),
            "change_ts": .double(ts),
        ])
    }

    private func handleClipboardSet(_ params: [String: MPValue]) -> MPValue {
        let text = params["text"]?.asString() ?? ""
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        return .map(["ok": .bool(true), "length": .int(Int64(text.count))])
    }

    private func handleClipboardLastChange() -> MPValue {
        let (_, count, ts) = clipboard.snapshot()
        return .map([
            "change_count": .int(Int64(count)),
            "change_ts": .double(ts),
        ])
    }

    // MARK: - Input

    private func handleInputTypeText(_ params: [String: MPValue]) -> MPValue {
        let text = params["text"]?.asString() ?? ""
        let method = params["method"]?.asString() ?? "paste"
        return input.typeText(text, method: method)
    }

    private func handleInputShortcut(_ params: [String: MPValue]) -> MPValue {
        let keys = params["keys"]?.asString() ?? ""
        return input.sendShortcut(keys)
    }

    // MARK: - Intents

    private func handleIntentDiscover(_ params: [String: MPValue]) -> MPValue {
        let appBundleId = params["bundle_id"]?.asString()
        return intentProvider.discover(appBundleId: appBundleId)
    }

    private func handleIntentPerform(_ params: [String: MPValue]) -> MPValue {
        let intentName = params["intent"]?.asString() ?? ""
        let intentParams = params["params"]?.asMap() ?? [:]
        return intentProvider.perform(intentName: intentName, params: intentParams)
    }

    // MARK: - Screen Capture

    private func handleScreenStartCapture(_ params: [String: MPValue]) throws -> MPValue {
        guard let provider = screenCaptureProvider else {
            throw HostRpcError(code: "screen_capture_not_available", message: "Screen capture provider not configured", details: [:])
        }
        let semaphore = DispatchSemaphore(value: 0)
        var result: MPValue = .map(["ok": .bool(false)])
        var captureError: Error?
        Task {
            do {
                try await provider.startCapture()
                result = .map(["ok": .bool(true)])
            } catch {
                captureError = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = captureError {
            throw HostRpcError(code: "screen_start_capture_failed", message: error.localizedDescription, details: [:])
        }
        return result
    }

    private func handleScreenStopCapture() throws -> MPValue {
        guard let provider = screenCaptureProvider else {
            throw HostRpcError(code: "screen_capture_not_available", message: "Screen capture provider not configured", details: [:])
        }
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            await provider.stopCapture()
            semaphore.signal()
        }
        semaphore.wait()
        return .map(["ok": .bool(true)])
    }

    private func handleScreenCaptureFrame(_ params: [String: MPValue]) throws -> MPValue {
        guard let provider = screenCaptureProvider else {
            throw HostRpcError(code: "screen_capture_not_available", message: "Screen capture provider not configured", details: [:])
        }
        guard provider.permissionGranted else {
            throw HostRpcError(
                code: "screen_capture_permission_denied",
                message: "Screen Recording permission not granted. Grant it in System Settings → Privacy & Security → Screen Recording, then restart OSHost.",
                details: [:]
            )
        }

        let bundleId = params["bundle_id"]?.asString()

        var region: CGRect? = nil
        if bundleId == nil, let regionMap = params["region"]?.asMap() {
            let x = regionMap["x"].flatMap(Self.extractDouble) ?? 0
            let y = regionMap["y"].flatMap(Self.extractDouble) ?? 0
            let w = regionMap["width"].flatMap(Self.extractDouble) ?? 0
            let h = regionMap["height"].flatMap(Self.extractDouble) ?? 0
            region = CGRect(x: x, y: y, width: w, height: h)
        }

        let isThumbnail = params["thumbnail"]?.asBool() ?? false
        let maxSize = params["max_size"].flatMap(Self.extractDouble).map { Int($0) }

        let mode: String
        if bundleId != nil { mode = "window" }
        else if region != nil { mode = "region" }
        else { mode = "fullscreen" }

        let semaphore = DispatchSemaphore(value: 0)
        var result: MPValue = .map(["ok": .bool(false)])
        var captureError: Error?
        Task {
            do {
                let data: Data
                if let bid = bundleId, !bid.isEmpty {
                    data = try await provider.captureFrame(bundleId: bid)
                } else {
                    data = try await provider.captureFrame(region: region)
                }
                let finalData: Data
                if isThumbnail, let limit = maxSize, limit > 0 {
                    finalData = Self.downsampleJPEG(data, maxDimension: limit)
                } else {
                    finalData = data
                }
                let base64 = finalData.base64EncodedString()
                let timestamp = Date().timeIntervalSince1970
                result = .map([
                    "frame_base64": .string(base64),
                    "timestamp": .double(timestamp),
                    "display_count": .int(Int64(provider.displayCount)),
                    "mode": .string(mode),
                ])
            } catch {
                captureError = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = captureError {
            throw HostRpcError(code: "screen_capture_frame_failed", message: String(describing: error), details: [:])
        }
        return result
    }

    private static func downsampleJPEG(_ data: Data, maxDimension: Int) -> Data {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return data }

        let w = cgImage.width
        let h = cgImage.height
        let longest = max(w, h)
        guard longest > maxDimension else { return data }

        let scale = CGFloat(maxDimension) / CGFloat(longest)
        let newW = Int(CGFloat(w) * scale)
        let newH = Int(CGFloat(h) * scale)

        guard let ctx = CGContext(
            data: nil, width: newW, height: newH,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return data }

        ctx.interpolationQuality = .low
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))
        guard let scaled = ctx.makeImage() else { return data }

        let rep = NSBitmapImageRep(cgImage: scaled)
        return rep.representation(using: .jpeg, properties: [.compressionFactor: 0.5]) ?? data
    }

    private func handleScreenPermissionStatus() -> MPValue {
        guard let provider = screenCaptureProvider else {
            return .map([
                "granted": .bool(false),
                "state": .string("not_available"),
            ])
        }
        let granted = provider.permissionGranted
        let state: String
        if granted {
            state = "granted"
        } else {
            state = "not_determined"
        }
        return .map([
            "granted": .bool(granted),
            "state": .string(state),
        ])
    }

    // MARK: - Recording Mode

    private func handleRecordingStart(_ params: [String: MPValue]) -> MPValue {
        let seq = recordingMode.start()
        fputs("OSHost: recording.start (high-fidelity mode active)\n", stderr)

        // Restart file watcher with lower latency
        if let legacyWatcher = fileWatcher as? LegacyFileWatcher {
            legacyWatcher.restartWithLatency(recordingMode.fsLatency)
        }

        // Push current focused app so the trajectory has an initial state
        appFocusMonitor?.pushCurrentFocus()

        audit.record(method: "recording.start", detail: [:])
        return .map([
            "ok": .bool(true),
            "sequence_start": .uint(seq),
        ])
    }

    private func handleRecordingStop() -> MPValue {
        let eventCount = recordingMode.stop()
        fputs("OSHost: recording.stop (event_count=\(eventCount))\n", stderr)

        // Restore normal latency
        if let legacyWatcher = fileWatcher as? LegacyFileWatcher {
            legacyWatcher.restartWithLatency(recordingMode.fsLatency)
        }

        audit.record(method: "recording.stop", detail: ["event_count": String(eventCount)])
        return .map([
            "ok": .bool(true),
            "event_count": .uint(eventCount),
        ])
    }

    // MARK: - Video Recording

    private func handleVideoStart(_ params: [String: MPValue]) throws -> MPValue {
        let outputDir = params["output_dir"]?.asString() ?? ""
        let fps = params["fps"].flatMap(Self.extractInt) ?? 5
        let resolutionScale = params["resolution_scale"].flatMap(Self.extractDouble) ?? 0.5
        let codec = params["codec"]?.asString() ?? "h264"
        let maxSegmentS = params["max_segment_s"].flatMap(Self.extractInt) ?? 600

        guard !outputDir.isEmpty else {
            throw HostRpcError(code: "invalid_params", message: "output_dir required", details: [:])
        }

        let semaphore = DispatchSemaphore(value: 0)
        var startError: Error?
        Task {
            do {
                try await self.videoService.start(
                    outputDir: URL(fileURLWithPath: outputDir),
                    fps: fps,
                    resolutionScale: resolutionScale,
                    codec: codec,
                    maxSegmentS: maxSegmentS
                )
            } catch {
                startError = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = startError {
            throw HostRpcError(code: "video_start_failed", message: String(describing: error), details: [:])
        }
        return .map(["ok": .bool(true)])
    }

    private func handleVideoStop() throws -> MPValue {
        let semaphore = DispatchSemaphore(value: 0)
        var segments: [VideoSegmentInfo] = []
        Task {
            segments = await self.videoService.stop()
            semaphore.signal()
        }
        semaphore.wait()
        let segmentArray: [MPValue] = segments.map { seg in
            .map([
                "path": .string(seg.path),
                "start_time": .double(seg.startTime),
                "duration_s": .double(seg.durationS),
                "fps": .double(seg.fps),
                "resolution": .array([.int(Int64(seg.width)), .int(Int64(seg.height))]),
                "codec": .string(seg.codec),
                "file_size": .int(Int64(seg.fileSize)),
            ])
        }
        return .map(["ok": .bool(true), "segments": .array(segmentArray)])
    }

    private func handleVideoPause() throws -> MPValue {
        videoService.pause()
        return .map(["ok": .bool(true)])
    }

    private func handleVideoResume() throws -> MPValue {
        videoService.resume()
        return .map(["ok": .bool(true)])
    }

    private func handleVideoExtractFrame(_ params: [String: MPValue]) throws -> MPValue {
        let videoPath = params["video_path"]?.asString() ?? ""
        let timestampS = params["timestamp_s"].flatMap(Self.extractDouble) ?? 0.0
        let maxSize = params["max_size"].flatMap(Self.extractInt)

        guard !videoPath.isEmpty else {
            throw HostRpcError(code: "invalid_params", message: "video_path required", details: [:])
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: MPValue = .map(["ok": .bool(false)])
        var extractError: Error?
        Task {
            do {
                let (data, width, height) = try await VideoFrameExtractor.extractFrame(
                    videoPath: videoPath,
                    timestampS: timestampS,
                    maxSize: maxSize
                )
                result = .map([
                    "ok": .bool(true),
                    "frame_base64": .string(data.base64EncodedString()),
                    "timestamp": .double(timestampS),
                    "width": .int(Int64(width)),
                    "height": .int(Int64(height)),
                ])
            } catch {
                extractError = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = extractError {
            throw HostRpcError(code: "video_extract_frame_failed", message: String(describing: error), details: [:])
        }
        return result
    }

    // MARK: - Helpers

    private static func extractInt(_ value: MPValue) -> Int? {
        switch value {
        case .int(let i): return Int(i)
        case .uint(let u): return Int(u)
        case .double(let d): return Int(d)
        default: return nil
        }
    }

    private static func extractDouble(_ value: MPValue) -> Double? {
        switch value {
        case .double(let d): return d
        case .int(let i): return Double(i)
        case .uint(let u): return Double(u)
        default: return nil
        }
    }
}

// MARK: - Supporting Types

struct HostRpcError: Error {
    let code: String
    let message: String
    let details: [String: MPValue]
}

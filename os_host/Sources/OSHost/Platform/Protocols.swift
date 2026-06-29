import CoreGraphics
import Foundation

// MARK: - Provider Protocols (ISP: each protocol covers one perception/execution domain)

protocol UISensorProvider {
    func readTree(bundleId: String?, preferIntents: Bool) -> MPValue
    func performAction(nodeId: String, action: String, params: [String: MPValue]) -> MPValue
}

extension UISensorProvider {
    func readTree(bundleId: String?) -> MPValue {
        readTree(bundleId: bundleId, preferIntents: false)
    }
}

protocol FileWatchProvider {
    func subscribe(path: String) -> String
    func recentEvents(limit: Int) -> [MPValue]
}

protocol FileOperationProvider {
    func listDirectory(path: String, includeHidden: Bool) throws -> MPValue
    func moveItem(src: String, dst: String) throws -> MPValue
    func copyItem(src: String, dst: String) throws -> MPValue
    func deleteItem(path: String) throws -> MPValue
}

protocol ClipboardProvider {
    func snapshot() -> (text: String, changeCount: Int, changeTs: TimeInterval)
}

protocol AppControlProvider {
    func launch(bundleId: String) -> Bool
    func activate(bundleId: String) -> Bool
    func listApps(filter: String, runningOnly: Bool) -> [MPValue]
}

protocol IntentProvider {
    var isAvailable: Bool { get }
    func discover(appBundleId: String?) -> MPValue
    func perform(intentName: String, params: [String: MPValue]) -> MPValue
}

struct ShellResult {
    let output: String
    let exitCode: Int32
}

protocol ShellProvider {
    func execute(command: String) throws -> String
    func executeWithStatus(command: String) throws -> ShellResult
}

extension ShellProvider {
    func execute(command: String) throws -> String {
        return try executeWithStatus(command: command).output
    }
}

protocol InputProvider {
    func typeText(_ text: String, method: String) -> MPValue
    func sendShortcut(_ keys: String) -> MPValue
}

protocol UIActionProvider {
    func startObserving() -> Bool
    func stopObserving()
    var isObserving: Bool { get }
}

/// Provider for screen capture capabilities (M3 Visual Track)
protocol ScreenCaptureProvider {
    /// Capture a single frame, optionally cropped to a region
    func captureFrame(region: CGRect?) async throws -> Data
    /// Capture the window of a specific app by bundle ID (display-independent)
    func captureFrame(bundleId: String) async throws -> Data
    /// Start continuous capture session
    func startCapture() async throws
    /// Stop capture session
    func stopCapture() async
    /// Whether a capture session is active
    var isCapturing: Bool { get }
    /// Whether screen recording permission is granted
    var permissionGranted: Bool { get }
    /// Number of connected displays
    var displayCount: Int { get }
}

extension ScreenCaptureProvider {
    func captureFrame(bundleId: String) async throws -> Data {
        return try await captureFrame(region: nil)
    }
    var displayCount: Int { 1 }
}

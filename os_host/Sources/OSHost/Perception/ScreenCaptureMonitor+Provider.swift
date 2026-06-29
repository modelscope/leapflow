import AppKit
import CoreGraphics
import Foundation

/// Adapter that bridges ScreenCaptureMonitor to the ScreenCaptureProvider protocol
/// required by RpcRouter, resolving the return-type mismatch (FrameData → Data)
/// and the sync/async mismatch on stopCapture.
final class ScreenCaptureProviderAdapter: ScreenCaptureProvider {
    private let monitor: ScreenCaptureMonitor

    init(_ monitor: ScreenCaptureMonitor) {
        self.monitor = monitor
    }

    func captureFrame(region: CGRect?) async throws -> Data {
        let frame: FrameData = try await monitor.captureFrame(region: region)
        return frame.data
    }

    func captureFrame(bundleId: String) async throws -> Data {
        let frame: FrameData = try await monitor.captureWindow(bundleId: bundleId)
        return frame.data
    }

    func startCapture() async throws {
        try await monitor.startCapture()
    }

    func stopCapture() async {
        monitor.stopCapture()
    }

    var isCapturing: Bool {
        monitor.isCapturing
    }

    var permissionGranted: Bool {
        monitor.permissionStatus() == .granted
    }

    var displayCount: Int {
        NSScreen.screens.count
    }
}

import AppKit
import Foundation

final class ClipboardMonitor: ClipboardProvider {
    private var lastSeenCount: Int
    private var lastChangeAt: TimeInterval
    private let pasteboard = NSPasteboard.general

    init() {
        lastSeenCount = pasteboard.changeCount
        lastChangeAt = Date().timeIntervalSince1970
    }

    /// Current text, pasteboard changeCount, and timestamp of last observed change.
    func snapshot() -> (text: String, changeCount: Int, changeTs: TimeInterval) {
        let c = pasteboard.changeCount
        if c != lastSeenCount {
            lastSeenCount = c
            lastChangeAt = Date().timeIntervalSince1970
        }
        let text = pasteboard.string(forType: .string) ?? ""
        return (text, c, lastChangeAt)
    }
}

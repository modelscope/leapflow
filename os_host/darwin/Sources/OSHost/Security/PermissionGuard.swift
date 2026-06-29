import AppKit
import ApplicationServices

enum PermissionGuard {
    /// Returns whether this process is trusted for accessibility control (AX).
    static func checkAccessibilityTrusted(prompt: Bool = false) -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let opts: NSDictionary = [key: prompt]
        return AXIsProcessTrustedWithOptions(opts)
    }
}

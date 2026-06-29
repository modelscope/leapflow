import AppKit
import Foundation

/// Monitors app activation via NSWorkspace and broadcasts `event.app_focus_change`
/// to all connected Python clients. Enables trajectory APP_SWITCH detection.
final class AppFocusMonitor {
    private let broadcaster: ClientBroadcaster
    private var observer: NSObjectProtocol?

    init(broadcaster: ClientBroadcaster) {
        self.broadcaster = broadcaster
    }

    func start() {
        observer = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            self?.handleActivation(notification)
        }
        fputs("AppFocusMonitor: started\n", stderr)
    }

    func stop() {
        if let obs = observer {
            NSWorkspace.shared.notificationCenter.removeObserver(obs)
            observer = nil
        }
    }

    /// Push the current frontmost app as an initial focus event (used by recording.start).
    func pushCurrentFocus() {
        guard let app = NSWorkspace.shared.frontmostApplication else { return }
        pushFocusEvent(app: app)
    }

    private func handleActivation(_ notification: Notification) {
        guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else {
            return
        }
        pushFocusEvent(app: app)
    }

    private func pushFocusEvent(app: NSRunningApplication) {
        let bundleId = app.bundleIdentifier ?? ""
        let appName = app.localizedName ?? bundleId
        let ts = Date().timeIntervalSince1970
        let title = windowTitle(for: app)

        var payload: [String: MPValue] = [
            "bundle_id": .string(bundleId),
            "app_name": .string(appName),
            "pid": .int(Int64(app.processIdentifier)),
            "ts": .double(ts),
        ]
        if !title.isEmpty {
            payload["window_title"] = .string(title)
        }

        let frame: [String: MPValue] = [
            "v": .int(1),
            "type": .string("event"),
            "event": .string("event.app_focus_change"),
            "payload": .map(payload),
            "ts": .double(ProcessInfo.processInfo.systemUptime),
        ]
        guard let data = try? MessageCodec.encodeFrame(frame) else { return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.broadcaster.broadcast(data: data)
        }
    }

    /// Read the main window title of an application via Accessibility API.
    private func windowTitle(for app: NSRunningApplication) -> String {
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        var windows: CFTypeRef?
        guard AXUIElementCopyAttributeValue(axApp, kAXWindowsAttribute as CFString, &windows) == .success,
              let arr = windows as? [AXUIElement],
              let mainWindow = arr.first else { return "" }

        var title: CFTypeRef?
        guard AXUIElementCopyAttributeValue(mainWindow, kAXTitleAttribute as CFString, &title) == .success,
              let t = title as? String else { return "" }
        return t
    }

    deinit { stop() }
}

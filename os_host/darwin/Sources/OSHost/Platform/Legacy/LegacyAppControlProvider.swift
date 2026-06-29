import AppKit
import Foundation

/// App lifecycle provider using NSWorkspace (all macOS versions).
struct LegacyAppControlProvider: AppControlProvider {
    func launch(bundleId: String) -> Bool {
        AppController.launch(bundleId: bundleId)
    }

    func activate(bundleId: String) -> Bool {
        AppController.activate(bundleId: bundleId)
    }

    func listApps(filter: String, runningOnly: Bool) -> [MPValue] {
        AppController.listApps(filter: filter, runningOnly: runningOnly)
    }
}

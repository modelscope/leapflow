import AppKit
import Foundation

enum AppController {
    @discardableResult
    static func launch(bundleId: String) -> Bool {
        guard let app = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleId) else {
            return false
        }
        let cfg = NSWorkspace.OpenConfiguration()
        cfg.promptsUserIfNeeded = false
        NSWorkspace.shared.openApplication(at: app, configuration: cfg, completionHandler: nil)
        return true
    }

    static func activate(bundleId: String) -> Bool {
        guard let running = NSRunningApplication.runningApplications(withBundleIdentifier: bundleId).first else {
            return false
        }
        return running.activate(options: [.activateAllWindows])
    }

    static func listApps(filter: String, runningOnly: Bool) -> [MPValue] {
        var apps: [MPValue] = []
        var seenBundleIds: Set<String> = []

        // 1. Running applications
        for app in NSWorkspace.shared.runningApplications {
            guard app.activationPolicy == .regular else { continue }
            let name = app.localizedName ?? ""
            let bid = app.bundleIdentifier ?? ""
            guard !bid.isEmpty else { continue }
            if !filter.isEmpty {
                let lower = filter.lowercased()
                guard name.lowercased().contains(lower) || bid.lowercased().contains(lower) else { continue }
            }
            seenBundleIds.insert(bid)
            apps.append(.map([
                "bundle_id": .string(bid),
                "name": .string(name),
                "running": .bool(true),
            ]))
        }

        if runningOnly { return apps }

        // 2. Installed applications from /Applications (recursive)
        let fm = FileManager.default
        let searchPaths = ["/Applications", "/System/Applications"]
        for searchPath in searchPaths {
            guard let enumerator = fm.enumerator(
                at: URL(fileURLWithPath: searchPath),
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { continue }

            for case let url as URL in enumerator {
                guard url.pathExtension == "app" else { continue }
                // Don't recurse into .app bundles
                enumerator.skipDescendants()

                guard let bundle = Bundle(url: url),
                      let bid = bundle.bundleIdentifier else { continue }
                guard !seenBundleIds.contains(bid) else { continue }

                let name = bundle.infoDictionary?["CFBundleName"] as? String
                    ?? bundle.infoDictionary?["CFBundleDisplayName"] as? String
                    ?? url.deletingPathExtension().lastPathComponent

                if !filter.isEmpty {
                    let lower = filter.lowercased()
                    guard name.lowercased().contains(lower) || bid.lowercased().contains(lower) else { continue }
                }

                seenBundleIds.insert(bid)
                apps.append(.map([
                    "bundle_id": .string(bid),
                    "name": .string(name),
                    "running": .bool(false),
                ]))
            }
        }

        return apps
    }
}

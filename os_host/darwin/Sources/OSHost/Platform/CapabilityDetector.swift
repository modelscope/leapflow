import Foundation

struct PlatformCapabilities {
    let platformId: String
    let osVersion: String
    var capabilities: Set<String>
    var metadata: [String: String]

    func toMPValue() -> MPValue {
        .map([
            "platform_id": .string(platformId),
            "os_version": .string(osVersion),
            "capabilities": .array(capabilities.sorted().map { .string($0) }),
            "metadata": .map(metadata.mapValues { .string($0) }),
        ])
    }
}

enum CapabilityDetector {
    static func detect() -> PlatformCapabilities {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        let versionStr = "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"

        var caps: Set<String> = [
            "fs.watch",
            "ax.tree_read",
            "ax.perform_action",
            "clipboard.read",
            "clipboard.watch",
            "file.ops",
            "app.launch",
            "app.activate",
            "shell.exec",
            "notification.send",
            "app.focus_watch",
            "recording.mode",
        ]

        var meta: [String: String] = [:]
        var platformId = "darwin_15"

        if #available(macOS 26.0, *) {
            caps.formUnion([
                "fs.semantic_index",
                "app_intents.discover",
                "app_intents.perform",
                "screen.capture_gpu",
            ])
            meta["intent_runtime"] = "tahoe_native"
            platformId = "darwin_26"
        }

        if PermissionGuard.checkAccessibilityTrusted(prompt: false) {
            meta["accessibility"] = "trusted"
        } else {
            caps.remove("ax.tree_read")
            caps.remove("ax.perform_action")
            meta["accessibility"] = "not_trusted"
        }

        return PlatformCapabilities(
            platformId: platformId,
            osVersion: versionStr,
            capabilities: caps,
            metadata: meta
        )
    }
}

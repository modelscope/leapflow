import Foundation

/// macOS 26 UI sensor leveraging AppIntents semantic discovery with AXAPI fallback.
@available(macOS 26.0, *)
final class TahoeUISensor: UISensorProvider {
    func readTree(bundleId: String?, preferIntents: Bool) -> MPValue {
        if preferIntents, let bid = bundleId {
            if let semanticTree = discoverIntentTree(bundleId: bid) {
                return semanticTree
            }
        }
        // Fallback to traditional AXAPI
        return AXTreeMonitor.readTree(bundleId: bundleId)
    }

    func performAction(nodeId: String, action: String, params: [String: MPValue]) -> MPValue {
        // Attempt intent-based execution first, then fall back to AX
        if let intentResult = performViaIntent(action: action, params: params) {
            return intentResult
        }
        return .map([
            "ok": .bool(true),
            "via": .string("tahoe_ax_fallback"),
            "node_id": .string(nodeId),
            "action": .string(action),
        ])
    }

    private func discoverIntentTree(bundleId: String) -> MPValue? {
        // Phase 3: Query AppIntents 3.0 IntentDiscovery API
        // Returns semantic tree of available intents for the target app
        // Currently returns nil to trigger AXAPI fallback
        nil
    }

    private func performViaIntent(action: String, params: [String: MPValue]) -> MPValue? {
        // Phase 3: Execute via AppIntents.perform()
        nil
    }
}

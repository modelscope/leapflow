import ApplicationServices
import AppKit
import Foundation

/// macOS 15 (and earlier) UI sensor using the traditional Accessibility API.
final class LegacyAXSensor: UISensorProvider {
    private let maxDepth = 6
    private let maxChildren = 40

    func readTree(bundleId: String?, preferIntents: Bool) -> MPValue {
        // preferIntents is ignored on Legacy — always use AXAPI
        return AXTreeMonitor.readTree(bundleId: bundleId)
    }

    func performAction(nodeId: String, action: String, params: [String: MPValue]) -> MPValue {
        let bundleId = params["bundle_id"]?.asString()

        guard !nodeId.isEmpty else {
            return .map(["ok": .bool(false), "error": .string("empty_node_id")])
        }

        guard let element = AXTreeMonitor.resolveElement(bundleId: bundleId, nodeId: nodeId) else {
            return .map([
                "ok": .bool(false),
                "error": .string("element_not_found"),
                "node_id": .string(nodeId),
            ])
        }

        let axAction = action.hasPrefix("AX") ? action : "AX\(action)"
        let result = AXUIElementPerformAction(element, axAction as CFString)

        if result == .success {
            return .map([
                "ok": .bool(true),
                "node_id": .string(nodeId),
                "action": .string(axAction),
            ])
        }

        return .map([
            "ok": .bool(false),
            "error": .string("action_failed"),
            "ax_error": .int(Int64(result.rawValue)),
            "node_id": .string(nodeId),
            "action": .string(axAction),
        ])
    }
}

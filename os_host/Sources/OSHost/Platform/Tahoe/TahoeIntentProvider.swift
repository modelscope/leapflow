import Foundation

/// macOS 26 AppIntents provider for discovering and executing app-exposed intents.
@available(macOS 26.0, *)
final class TahoeIntentProvider: IntentProvider {
    var isAvailable: Bool { true }

    func discover(appBundleId: String?) -> MPValue {
        // Phase 3: Use AppIntents 3.0 DiscoveryService to enumerate
        // available intents for the specified app (or all apps)
        //
        // Expected output: array of intent descriptors with:
        //   - name, description, parameters, return_type
        return .map([
            "ok": .bool(true),
            "stub": .bool(true),
            "intents": .array([]),
            "hint": .string("Intent discovery will be implemented with macOS 26 SDK"),
        ])
    }

    func perform(intentName: String, params: [String: MPValue]) -> MPValue {
        // Phase 3: Instantiate the named Intent, populate parameters,
        // and call perform() via the AppIntents runtime
        return .map([
            "ok": .bool(true),
            "stub": .bool(true),
            "intent": .string(intentName),
            "hint": .string("Intent execution will be implemented with macOS 26 SDK"),
        ])
    }
}

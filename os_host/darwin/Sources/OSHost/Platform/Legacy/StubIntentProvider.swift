import Foundation

/// Stub intent provider for macOS versions without AppIntents runtime.
struct StubIntentProvider: IntentProvider {
    var isAvailable: Bool { false }

    func discover(appBundleId: String?) -> MPValue {
        .map([
            "ok": .bool(false),
            "error": .string("intents_not_available"),
            "hint": .string("AppIntents requires macOS 26 (Tahoe) or later."),
        ])
    }

    func perform(intentName: String, params: [String: MPValue]) -> MPValue {
        .map([
            "ok": .bool(false),
            "error": .string("intents_not_available"),
            "intent": .string(intentName),
        ])
    }
}

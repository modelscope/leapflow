import Foundation

/// MVP placeholder for App Intents–based execution (real automation comes later).
enum IntentExecutor {
    static func runStub(name: String, parameters: [String: MPValue]) -> MPValue {
        .map([
            "ok": .bool(true),
            "stub": .bool(true),
            "intent": .string(name),
            "parameters": .map(parameters),
        ])
    }
}

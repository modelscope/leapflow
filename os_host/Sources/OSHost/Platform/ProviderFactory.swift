import Foundation

/// Factory that constructs the appropriate provider implementations based on runtime capabilities.
/// Uses #available for compile-time safety and runtime capability detection.
enum ProviderFactory {
    static func makeUISensor() -> UISensorProvider {
        if #available(macOS 26.0, *) {
            return TahoeUISensor()
        }
        return LegacyAXSensor()
    }

    static func makeFileWatcher(onEvent: @escaping (String, UInt32, TimeInterval) -> Void) -> FileWatchProvider {
        if #available(macOS 26.0, *) {
            return TahoeFileWatcher(onEvent: onEvent)
        }
        return LegacyFileWatcher(onEvent: onEvent)
    }

    static func makeFileOperator() -> FileOperationProvider {
        return LegacyFileOperationProvider()
    }

    static func makeClipboard() -> ClipboardProvider {
        return ClipboardMonitor()
    }

    static func makeAppControl() -> AppControlProvider {
        return LegacyAppControlProvider()
    }

    static func makeIntentProvider() -> IntentProvider {
        if #available(macOS 26.0, *) {
            return TahoeIntentProvider()
        }
        return StubIntentProvider()
    }

    static func makeShellProvider() -> ShellProvider {
        return ZshShellProvider()
    }

    static func makeUIActionProvider(broadcaster: ClientBroadcaster) -> UIActionProvider {
        return UIActionObserver(broadcaster: broadcaster)
    }

    static func makeInputProvider() -> InputProvider {
        return InputInjector()
    }

    static func makeScreenCaptureProvider(broadcaster: ClientBroadcaster) -> ScreenCaptureProvider? {
        if #available(macOS 14.0, *) {
            return ScreenCaptureProviderAdapter(ScreenCaptureMonitor(broadcaster: broadcaster))
        }
        return nil
    }
}

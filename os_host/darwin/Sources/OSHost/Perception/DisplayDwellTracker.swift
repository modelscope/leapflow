import AppKit
import Foundation
import ScreenCaptureKit

/// Configuration for mouse-driven display tracking.
struct MouseTrackingConfig {
    /// Sliding window duration (seconds) for dwell calculation
    var dwellWindowDuration: Double = 5.0
    /// Threshold (0-1): display is "active" when dwell ratio >= this value
    var dwellThreshold: Double = 0.70
    /// Minimum interval between display switches (seconds)
    var minSwitchInterval: Double = 3.0
    /// Background polling frequency (Hz) for mouse position sampling
    var pollingFrequency: Double = 1.0
    /// Whether dynamic display switching is enabled
    var displaySwitchEnabled: Bool = true
}

/// Tracks mouse dwell time across displays using a sliding window.
final class DisplayDwellTracker {
    private let config: MouseTrackingConfig
    private var samples: [(displayID: UInt32, timestamp: TimeInterval)] = []
    private var lastSwitchTime: TimeInterval = 0
    private(set) var currentActiveDisplayID: UInt32?

    init(config: MouseTrackingConfig = MouseTrackingConfig()) {
        self.config = config
    }

    /// Record a mouse position sample mapped to a display ID.
    func recordSample(displayID: UInt32) {
        let now = ProcessInfo.processInfo.systemUptime
        samples.append((displayID: displayID, timestamp: now))
        // Trim samples outside the sliding window
        let cutoff = now - config.dwellWindowDuration
        samples.removeAll { $0.timestamp < cutoff }
    }

    /// Detect if active display should change.
    /// Returns the new display ID if a switch is warranted, nil otherwise.
    func detectActiveDisplay() -> UInt32? {
        guard config.displaySwitchEnabled, !samples.isEmpty else { return nil }

        let now = ProcessInfo.processInfo.systemUptime
        // Enforce minimum switch interval
        if now - lastSwitchTime < config.minSwitchInterval {
            return nil
        }

        // Calculate dwell ratio per display
        var counts: [UInt32: Int] = [:]
        for sample in samples {
            counts[sample.displayID, default: 0] += 1
        }

        let total = samples.count
        guard total > 0 else { return nil }

        // Find display with highest dwell ratio
        guard let (topDisplayID, topCount) = counts.max(by: { $0.value < $1.value }) else {
            return nil
        }

        let ratio = Double(topCount) / Double(total)

        // Only switch if above threshold AND different from current
        if ratio >= config.dwellThreshold && topDisplayID != currentActiveDisplayID {
            currentActiveDisplayID = topDisplayID
            lastSwitchTime = now
            return topDisplayID
        }

        return nil
    }

    /// Set the initial active display (called at recording start).
    func setInitialDisplay(_ displayID: UInt32) {
        currentActiveDisplayID = displayID
        lastSwitchTime = ProcessInfo.processInfo.systemUptime
    }

    /// Reset tracker state.
    func reset() {
        samples.removeAll()
        currentActiveDisplayID = nil
        lastSwitchTime = 0
    }

    /// Map a mouse position to a display ID.
    static func displayIDForMousePosition(_ mousePos: CGPoint, displays: [SCDisplay]) -> UInt32? {
        for display in displays {
            if display.frame.contains(mousePos) {
                return UInt32(display.displayID)
            }
        }
        return nil
    }
}

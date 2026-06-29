import Foundation

/// Centralized recording state shared across perception providers.
/// When active, providers use tighter latency/coalesce parameters for higher-fidelity capture.
final class RecordingMode {
    private let lock = NSLock()
    private var _active = false
    private var _sequenceCounter: UInt64 = 0

    var isActive: Bool {
        lock.lock(); defer { lock.unlock() }
        return _active
    }

    /// FSEvents delivery latency: 0.1s during recording, 0.35s idle.
    var fsLatency: TimeInterval { isActive ? 0.1 : 0.35 }

    /// FS event coalesce window: 0.3s during recording, 1.0s idle.
    var fsCoalesceWindow: TimeInterval { isActive ? 0.3 : 1.0 }

    @discardableResult
    func start() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        _active = true
        _sequenceCounter = 0
        return _sequenceCounter
    }

    @discardableResult
    func stop() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        _active = false
        let final = _sequenceCounter
        _sequenceCounter = 0
        return final
    }

    func nextSequence() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        _sequenceCounter += 1
        return _sequenceCounter
    }
}

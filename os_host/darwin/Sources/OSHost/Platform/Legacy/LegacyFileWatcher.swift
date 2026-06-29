import CoreServices
import Foundation

/// macOS 15 file watcher backed by FSEvents (wraps existing FSEventsWatcher).
final class LegacyFileWatcher: FileWatchProvider {
    private let lock = NSLock()
    private var subscribedPaths: Set<String> = []
    private var eventLog: [(path: String, flags: UInt32, ts: TimeInterval)] = []
    private var watcher: FSEventsWatcher
    private var currentLatency: TimeInterval = 0.35

    init(onEvent: @escaping (String, UInt32, TimeInterval) -> Void) {
        let externalCallback = onEvent
        self.watcher = FSEventsWatcher { _, _, _ in }
        self.watcher = FSEventsWatcher { [weak self] path, flags, ts in
            self?.record(path: path, flags: flags, ts: ts)
            externalCallback(path, flags, ts)
        }
    }

    func subscribe(path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        lock.lock()
        subscribedPaths.insert(expanded)
        let paths = Array(subscribedPaths)
        lock.unlock()
        watcher.start(paths: paths, latency: currentLatency)
        return UUID().uuidString
    }

    func recentEvents(limit: Int) -> [MPValue] {
        lock.lock()
        let snap = Array(eventLog.suffix(limit))
        lock.unlock()
        return snap.map { e in
            .map([
                "path": .string(e.path),
                "flags": .uint(UInt64(e.flags)),
                "ts": .double(e.ts),
            ])
        }
    }

    /// Restart the underlying FSEvents stream with a new latency (used by recording mode).
    func restartWithLatency(_ latency: TimeInterval) {
        lock.lock()
        currentLatency = latency
        lock.unlock()
        watcher.restart(latency: latency)
    }

    private static let coalesceWindow: TimeInterval = 1.0

    private func record(path: String, flags: UInt32, ts: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        if let last = eventLog.last, last.path == path,
           ts - last.ts < Self.coalesceWindow {
            eventLog[eventLog.count - 1] = (path, last.flags | flags, ts)
            return
        }
        eventLog.append((path, flags, ts))
        if eventLog.count > 500 {
            eventLog.removeFirst(eventLog.count - 500)
        }
    }
}

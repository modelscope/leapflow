import CoreServices
import Foundation

/// macOS 26 file watcher with Semantic Index integration on top of FSEvents.
@available(macOS 26.0, *)
final class TahoeFileWatcher: FileWatchProvider {
    private let legacyWatcher: LegacyFileWatcher

    init(onEvent: @escaping (String, UInt32, TimeInterval) -> Void) {
        self.legacyWatcher = LegacyFileWatcher(onEvent: onEvent)
        // Phase 3: Additionally subscribe to Tahoe Semantic Index notifications
        // to receive content-level change signals (e.g. "document topic changed")
    }

    func subscribe(path: String) -> String {
        let subId = legacyWatcher.subscribe(path: path)
        // Phase 3: Also register with SemanticIndexService for this path
        return subId
    }

    func recentEvents(limit: Int) -> [MPValue] {
        // Merge FSEvents with Semantic Index events
        let fsEvents = legacyWatcher.recentEvents(limit: limit)
        // Phase 3: Append semantic index change events
        return fsEvents
    }
}

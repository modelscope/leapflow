import CoreServices
import Foundation

final class FSEventsWatcher {
    private var stream: FSEventStreamRef?
    private let onEvent: (String, FSEventStreamEventFlags, TimeInterval) -> Void
    private var currentLatency: CFTimeInterval = 0.35
    private var currentPaths: [String] = []

    init(onEvent: @escaping (String, FSEventStreamEventFlags, TimeInterval) -> Void) {
        self.onEvent = onEvent
    }

    func start(paths: [String], latency: CFTimeInterval = 0.35) {
        stop()
        guard !paths.isEmpty else { return }
        currentPaths = paths
        currentLatency = latency

        var ctx = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passUnretained(self).toOpaque(),
            retain: nil,
            release: nil,
            copyDescription: nil
        )

        let cb: FSEventStreamCallback = { _, clientInfo, numEvents, eventPaths, eventFlags, _ in
            guard let clientInfo else { return }
            let watcher = Unmanaged<FSEventsWatcher>.fromOpaque(clientInfo).takeUnretainedValue()
            let pathsArray = unsafeBitCast(eventPaths, to: NSArray.self) as! [String]
            for i in 0 ..< numEvents {
                let p = pathsArray[Int(i)]
                let flags = eventFlags[Int(i)]
                if watcher.shouldIgnore(path: p) { continue }
                watcher.onEvent(p, flags, Date().timeIntervalSince1970)
            }
        }

        let cfPaths = paths as CFArray
        let since = FSEventStreamEventId(UInt64(kFSEventStreamEventIdSinceNow))
        let flags = FSEventStreamCreateFlags(
            kFSEventStreamCreateFlagFileEvents
                | kFSEventStreamCreateFlagUseCFTypes
                | kFSEventStreamCreateFlagNoDefer
        )

        stream = FSEventStreamCreate(kCFAllocatorDefault, cb, &ctx, cfPaths, since, latency, flags)
        guard let stream else { return }
        FSEventStreamSetDispatchQueue(stream, DispatchQueue.global(qos: .utility))
        FSEventStreamStart(stream)
    }

    func restart(latency: CFTimeInterval) {
        guard !currentPaths.isEmpty else { return }
        start(paths: currentPaths, latency: latency)
    }

    func stop() {
        guard let stream else { return }
        FSEventStreamStop(stream)
        FSEventStreamInvalidate(stream)
        FSEventStreamRelease(stream)
        self.stream = nil
    }

    deinit {
        stop()
    }

    private static let ignoredDirComponents: [String] = [
        "/library/", "/.git/", "/.tmp/", "/.leapflow/", "/.leap/",
    ]

    private static let ignoredSuffixes: [String] = [
        "/.ds_store", ".sqlite-journal", ".sqlite-wal", ".sqlite-shm",
    ]

    private func shouldIgnore(path: String) -> Bool {
        let l = path.lowercased()
        for dir in Self.ignoredDirComponents {
            if l.contains(dir) { return true }
        }
        for suffix in Self.ignoredSuffixes {
            if l.hasSuffix(suffix) { return true }
        }
        return false
    }
}

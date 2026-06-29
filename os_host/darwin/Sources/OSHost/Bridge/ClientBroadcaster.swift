import Darwin
import Foundation

/// Thread-safe manager for connected client file descriptors.
/// Enables full-duplex communication: event frames can be pushed to clients
/// while RPC responses flow on the same connection.
final class ClientBroadcaster {
    private let lock = NSLock()
    private var clients: [Int32: NSLock] = [:]

    func addClient(_ fd: Int32) {
        lock.lock(); defer { lock.unlock() }
        clients[fd] = NSLock()
    }

    func removeClient(_ fd: Int32) {
        lock.lock(); defer { lock.unlock() }
        clients.removeValue(forKey: fd)
    }

    /// Per-fd write lock. The RPC response path must acquire this before writing.
    func writeLock(for fd: Int32) -> NSLock? {
        lock.lock(); defer { lock.unlock() }
        return clients[fd]
    }

    /// Broadcast an event frame to all connected clients (best-effort).
    func broadcast(data: Data) {
        lock.lock()
        let snapshot = clients
        lock.unlock()

        var dead: [Int32] = []
        for (fd, wLock) in snapshot {
            wLock.lock()
            let ok = writeAll(fd: fd, data: data)
            wLock.unlock()
            if !ok { dead.append(fd) }
        }

        if !dead.isEmpty {
            lock.lock()
            for fd in dead { clients.removeValue(forKey: fd) }
            lock.unlock()
        }
    }

    private func writeAll(fd: Int32, data: Data) -> Bool {
        data.withUnsafeBytes { raw -> Bool in
            guard let ptr = raw.bindMemory(to: UInt8.self).baseAddress else { return false }
            var sent = 0
            let total = data.count
            while sent < total {
                let n = write(fd, ptr + sent, total - sent)
                if n <= 0 { return false }
                sent += n
            }
            return true
        }
    }
}

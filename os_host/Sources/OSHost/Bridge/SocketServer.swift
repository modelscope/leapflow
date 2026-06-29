import Darwin
import Foundation

enum SocketServer {
    static func serve(socketPath: String, router: RpcRouter, broadcaster: ClientBroadcaster) {
        let path = (socketPath as NSString).expandingTildeInPath
        unlink(path)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            fputs("OSHost: socket() failed\n", stderr)
            return
        }
        var closeFd = true
        defer {
            if closeFd { close(fd) }
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path)
        path.withCString { cstr in
            withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
                let raw = UnsafeMutableRawPointer(ptr)
                strlcpy(raw.assumingMemoryBound(to: CChar.self), cstr, maxLen)
            }
        }

        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let bindErr = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                bind(fd, sa, len)
            }
        }
        guard bindErr == 0 else {
            perror("OSHost: bind")
            return
        }

        guard listen(fd, 16) == 0 else {
            perror("OSHost: listen")
            return
        }

        closeFd = false
        fputs("OSHost listening on \(path)\n", stderr)

        while true {
            var clientAddr = sockaddr_un()
            var clientLen = socklen_t(MemoryLayout<sockaddr_un>.size)
            let client = withUnsafeMutablePointer(to: &clientAddr) { ptr -> Int32 in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    accept(fd, sa, &clientLen)
                }
            }
            guard client >= 0 else {
                perror("OSHost: accept")
                continue
            }
            fputs("OSHost: client connected (fd=\(client))\n", stderr)
            DispatchQueue.global(qos: .userInitiated).async {
                broadcaster.addClient(client)
                defer {
                    fputs("OSHost: client disconnected (fd=\(client))\n", stderr)
                    broadcaster.removeClient(client); close(client)
                }
                Self.handleClient(fd: client, router: router, broadcaster: broadcaster)
            }
        }
    }

    private static let quietMethods: Set<String> = [
        "screen.capture_frame", "ping",
    ]

    private static func handleClient(fd: Int32, router: RpcRouter, broadcaster: ClientBroadcaster) {
        do {
            while true {
                let lenData = try readExact(fd: fd, count: 4)
                let bodyLen = Int(UInt32(lenData[0]) << 24 | UInt32(lenData[1]) << 16 | UInt32(lenData[2]) << 8 | UInt32(lenData[3]))
                guard bodyLen > 0, bodyLen < 32 * 1024 * 1024 else {
                    fputs("OSHost: invalid frame size \(bodyLen) from fd=\(fd)\n", stderr)
                    return
                }
                let body = try readExact(fd: fd, count: bodyLen)
                let req = try MessageCodec.decodeFrame(body)
                let method = req["method"]?.asString() ?? "?"
                let rid = req["id"]?.asString() ?? "?"
                let verbose = !quietMethods.contains(method)
                if verbose {
                    fputs("OSHost: RPC \(method) (id=\(rid.prefix(8))) from fd=\(fd)\n", stderr)
                }
                let resp = try router.handleRpc(frame: req)
                let wLock = broadcaster.writeLock(for: fd)
                wLock?.lock()
                defer { wLock?.unlock() }
                try writeAll(fd: fd, data: resp)
                if verbose {
                    fputs("OSHost: RPC \(method) (id=\(rid.prefix(8))) → ok\n", stderr)
                }
            }
        } catch {
            fputs("OSHost: client fd=\(fd) loop ended: \(error)\n", stderr)
        }
    }

    private static func readExact(fd: Int32, count: Int) throws -> Data {
        var buffer = [UInt8](repeating: 0, count: count)
        var received = 0
        while received < count {
            let n = read(fd, &buffer[received], count - received)
            if n == 0 {
                throw URLError(.cannotLoadFromNetwork)
            }
            if n < 0 {
                throw URLError(.cannotLoadFromNetwork)
            }
            received += n
        }
        return Data(buffer)
    }

    private static func writeAll(fd: Int32, data: Data) throws {
        try data.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: UInt8.self).baseAddress!
            var sent = 0
            let total = data.count
            while sent < total {
                let n = write(fd, ptr + sent, total - sent)
                if n <= 0 { throw URLError(.cannotWriteToFile) }
                sent += n
            }
        }
    }
}

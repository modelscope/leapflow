import Foundation

/// Append-only JSON Lines audit sink (best-effort).
final class AuditLog {
    private let queue = DispatchQueue(label: "leapflow.audit", qos: .utility)
    private let url: URL

    init() {
        let base = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".leapflow", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        url = base.appendingPathComponent("oshost_audit.jsonl")
    }

    func record(method: String, detail: [String: Any]) {
        var row = detail
        row["method"] = method
        row["ts"] = ISO8601DateFormatter().string(from: Date())
        guard let data = try? JSONSerialization.data(withJSONObject: row),
              let line = String(data: data, encoding: .utf8)
        else { return }
        let payload = (line + "\n").data(using: .utf8) ?? Data()
        queue.async {
            if !FileManager.default.fileExists(atPath: self.url.path) {
                FileManager.default.createFile(atPath: self.url.path, contents: nil)
            }
            do {
                let h = try FileHandle(forWritingTo: self.url)
                defer { try? h.close() }
                try h.seekToEnd()
                try h.write(contentsOf: payload)
            } catch {
                try? payload.write(to: self.url)
            }
        }
    }
}

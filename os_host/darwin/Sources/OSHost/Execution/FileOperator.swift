import Foundation

enum FileOperatorError: Error {
    case notFound
    case exists
}

enum FileOperator {
    static func listDirectory(path: String, includeHidden: Bool = false) throws -> [MPValue] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        var opts: FileManager.DirectoryEnumerationOptions = [.skipsPackageDescendants]
        if !includeHidden {
            opts.insert(.skipsHiddenFiles)
        }
        let fm = FileManager.default
        let items = try fm.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey, .contentModificationDateKey],
            options: opts
        )
        return try items.map { u in
            let vals = try u.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey, .contentModificationDateKey])
            return .map([
                "path": .string(u.path),
                "name": .string(u.lastPathComponent),
                "is_dir": .bool(vals.isDirectory ?? false),
                "size": .int(Int64(vals.fileSize ?? 0)),
                "mtime": .double(vals.contentModificationDate?.timeIntervalSince1970 ?? 0),
            ])
        }
    }

    static func moveItem(src: String, dst: String) throws {
        let fm = FileManager.default
        let s = URL(fileURLWithPath: (src as NSString).expandingTildeInPath)
        var d = URL(fileURLWithPath: (dst as NSString).expandingTildeInPath)
        try fm.createDirectory(at: d.deletingLastPathComponent(), withIntermediateDirectories: true)
        if d.hasDirectoryPath {
            d = d.appendingPathComponent(s.lastPathComponent)
        }
        if fm.fileExists(atPath: d.path) {
            try fm.removeItem(at: d)
        }
        try fm.moveItem(at: s, to: d)
    }

    static func copyItem(src: String, dst: String) throws {
        let fm = FileManager.default
        let s = URL(fileURLWithPath: (src as NSString).expandingTildeInPath)
        let d = URL(fileURLWithPath: (dst as NSString).expandingTildeInPath)
        try fm.createDirectory(at: d.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: d.path) {
            try fm.removeItem(at: d)
        }
        try fm.copyItem(at: s, to: d)
    }

    static func delete(path: String) throws {
        let fm = FileManager.default
        let p = (path as NSString).expandingTildeInPath
        if fm.fileExists(atPath: p) {
            try fm.removeItem(atPath: p)
        }
    }
}

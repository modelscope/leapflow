import Foundation

/// File operations provider using Foundation FileManager (all macOS versions).
struct LegacyFileOperationProvider: FileOperationProvider {
    func listDirectory(path: String, includeHidden: Bool) throws -> MPValue {
        let entries = try FileOperator.listDirectory(path: path, includeHidden: includeHidden)
        return .map([
            "path": .string((path as NSString).expandingTildeInPath),
            "entries": .array(entries),
        ])
    }

    func moveItem(src: String, dst: String) throws -> MPValue {
        try FileOperator.moveItem(src: src, dst: dst)
        return .map(["ok": .bool(true), "dst": .string(dst)])
    }

    func copyItem(src: String, dst: String) throws -> MPValue {
        try FileOperator.copyItem(src: src, dst: dst)
        return .map(["ok": .bool(true), "dst": .string(dst)])
    }

    func deleteItem(path: String) throws -> MPValue {
        try FileOperator.delete(path: path)
        return .map(["ok": .bool(true), "path": .string(path)])
    }
}

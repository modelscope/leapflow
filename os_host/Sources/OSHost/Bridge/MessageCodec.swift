import Foundation

/// Minimal MessagePack encode/decode aligned with Python `msgpack` (`use_bin_type=True`, string keys).
enum MessageCodecError: Error {
    case truncated
    case invalidFormat
    case unsupported
}

enum MPValue: Equatable {
    case null
    case bool(Bool)
    case int(Int64)
    case uint(UInt64)
    case double(Double)
    case string(String)
    case binary(Data)
    case array([MPValue])
    case map([String: MPValue])
}

private struct MPReader {
    let bytes: [UInt8]
    var offset: Int = 0

    mutating func need(_ n: Int) throws {
        if offset + n > bytes.count { throw MessageCodecError.truncated }
    }

    mutating func readU8() throws -> UInt8 {
        try need(1)
        let v = bytes[offset]
        offset += 1
        return v
    }

    mutating func readU16BE() throws -> UInt16 {
        try need(2)
        let v = UInt16(bytes[offset]) << 8 | UInt16(bytes[offset + 1])
        offset += 2
        return v
    }

    mutating func readU32BE() throws -> UInt32 {
        try need(4)
        let v =
            (UInt32(bytes[offset]) << 24) | (UInt32(bytes[offset + 1]) << 16)
            | (UInt32(bytes[offset + 2]) << 8) | UInt32(bytes[offset + 3])
        offset += 4
        return v
    }

    mutating func readU64BE() throws -> UInt64 {
        try need(8)
        var v: UInt64 = 0
        for i in 0 ..< 8 {
            v = (v << 8) | UInt64(bytes[offset + i])
        }
        offset += 8
        return v
    }

    mutating func readData(_ n: Int) throws -> Data {
        try need(n)
        let d = Data(bytes[offset ..< offset + n])
        offset += n
        return d
    }

    mutating func unpack() throws -> MPValue {
        let b = try readU8()
        switch b {
        case 0xc0:
            return .null
        case 0xc2:
            return .bool(false)
        case 0xc3:
            return .bool(true)
        case 0xcc:
            return .uint(UInt64(try readU8()))
        case 0xcd:
            return .uint(UInt64(try readU16BE()))
        case 0xce:
            return .uint(UInt64(try readU32BE()))
        case 0xcf:
            return .uint(try readU64BE())
        case 0xd0:
            return .int(Int64(Int8(bitPattern: try readU8())))
        case 0xd1:
            return .int(Int64(Int16(bitPattern: try readU16BE())))
        case 0xd2:
            return .int(Int64(Int32(bitPattern: try readU32BE())))
        case 0xd3:
            return .int(Int64(bitPattern: try readU64BE()))
        case 0xca:
            let u = try readU32BE()
            return .double(Double(Float(bitPattern: u)))
        case 0xcb:
            let u = try readU64BE()
            return .double(Double(bitPattern: u))
        case 0xd9:
            let n = Int(try readU8())
            let data = try readData(n)
            guard let s = String(data: data, encoding: .utf8) else { throw MessageCodecError.invalidFormat }
            return .string(s)
        case 0xda:
            let n = Int(try readU16BE())
            let data = try readData(n)
            guard let s = String(data: data, encoding: .utf8) else { throw MessageCodecError.invalidFormat }
            return .string(s)
        case 0xdb:
            let n = Int(try readU32BE())
            let data = try readData(n)
            guard let s = String(data: data, encoding: .utf8) else { throw MessageCodecError.invalidFormat }
            return .string(s)
        case 0xc4:
            let n = Int(try readU8())
            return .binary(try readData(n))
        case 0xc5:
            let n = Int(try readU16BE())
            return .binary(try readData(n))
        case 0xc6:
            let n = Int(try readU32BE())
            return .binary(try readData(n))
        default:
            if b >= 0x00, b <= 0x7f {
                return .uint(UInt64(b))
            }
            if b >= 0xe0 {
                return .int(Int64(Int8(bitPattern: b)))
            }
            if b >= 0xa0, b <= 0xbf {
                let n = Int(b & 0x1f)
                let data = try readData(n)
                guard let s = String(data: data, encoding: .utf8) else { throw MessageCodecError.invalidFormat
                }
                return .string(s)
            }
            if b >= 0x90, b <= 0x9f {
                let n = Int(b & 0x0f)
                var arr: [MPValue] = []
                arr.reserveCapacity(n)
                for _ in 0 ..< n { arr.append(try unpack()) }
                return .array(arr)
            }
            if b >= 0x80, b <= 0x8f {
                let n = Int(b & 0x0f)
                var m: [String: MPValue] = [:]
                for _ in 0 ..< n {
                    let k = try unpack()
                    let val = try unpack()
                    guard case .string(let ks) = k else { throw MessageCodecError.invalidFormat }
                    m[ks] = val
                }
                return .map(m)
            }
            if b == 0xdc {
                let n = Int(try readU16BE())
                var arr: [MPValue] = []
                arr.reserveCapacity(n)
                for _ in 0 ..< n { arr.append(try unpack()) }
                return .array(arr)
            }
            if b == 0xdd {
                let n = Int(try readU32BE())
                var arr: [MPValue] = []
                for _ in 0 ..< n { arr.append(try unpack()) }
                return .array(arr)
            }
            if b == 0xde {
                let n = Int(try readU16BE())
                var m: [String: MPValue] = [:]
                for _ in 0 ..< n {
                    let k = try unpack()
                    let val = try unpack()
                    guard case .string(let ks) = k else { throw MessageCodecError.invalidFormat }
                    m[ks] = val
                }
                return .map(m)
            }
            if b == 0xdf {
                let n = Int(try readU32BE())
                var m: [String: MPValue] = [:]
                for _ in 0 ..< n {
                    let k = try unpack()
                    let val = try unpack()
                    guard case .string(let ks) = k else { throw MessageCodecError.invalidFormat }
                    m[ks] = val
                }
                return .map(m)
            }
            throw MessageCodecError.unsupported
        }
    }
}

private struct MPWriter {
    var data = Data()

    mutating func writeU8(_ v: UInt8) {
        data.append(v)
    }

    mutating func writeU32BE(_ v: UInt32) {
        var be = v.bigEndian
        Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
    }

    mutating func writeU64BE(_ v: UInt64) {
        var be = v.bigEndian
        Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
    }

    mutating func pack(_ v: MPValue) {
        switch v {
        case .null:
            writeU8(0xc0)
        case .bool(let b):
            writeU8(b ? 0xc3 : 0xc2)
        case .int(let i):
            if i >= 0, i <= 127 {
                writeU8(UInt8(truncatingIfNeeded: i))
            } else if i >= -32, i < 0 {
                writeU8(UInt8(bitPattern: Int8(i)))
            } else if i >= Int32.min, i <= Int32.max {
                writeU8(0xd2)
                var be = UInt32(bitPattern: Int32(i)).bigEndian
                Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
            } else {
                writeU8(0xd3)
                var be = UInt64(bitPattern: i).bigEndian
                Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
            }
        case .uint(let u) where u <= 127:
            writeU8(UInt8(u))
        case .uint(let u) where u <= UInt64(UInt16.max):
            writeU8(0xcd)
            var be = UInt16(truncatingIfNeeded: u).bigEndian
            Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
        case .uint(let u) where u <= UInt64(UInt32.max):
            writeU8(0xce)
            writeU32BE(UInt32(truncatingIfNeeded: u))
        case .uint(let u):
            writeU8(0xcf)
            writeU64BE(u)
        case .double(let d):
            writeU8(0xcb)
            var bits = d.bitPattern.bigEndian
            Swift.withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        case .string(let s):
            let utf8 = Data(s.utf8)
            let n = utf8.count
            if n < 32 {
                writeU8(0xa0 | UInt8(n))
            } else if n <= Int(UInt8.max) {
                writeU8(0xd9)
                writeU8(UInt8(n))
            } else if n <= Int(UInt16.max) {
                writeU8(0xda)
                var be = UInt16(n).bigEndian
                Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
            } else {
                writeU8(0xdb)
                writeU32BE(UInt32(n))
            }
            data.append(utf8)
        case .binary(let b):
            let n = b.count
            if n <= Int(UInt8.max) {
                writeU8(0xc4)
                writeU8(UInt8(n))
            } else if n <= Int(UInt16.max) {
                writeU8(0xc5)
                var be = UInt16(n).bigEndian
                Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
            } else {
                writeU8(0xc6)
                writeU32BE(UInt32(n))
            }
            data.append(b)
        case .array(let arr):
            let n = arr.count
            if n < 16 {
                writeU8(0x90 | UInt8(n))
            } else if n <= Int(UInt16.max) {
                writeU8(0xdc)
                var be = UInt16(n).bigEndian
                Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
            } else {
                writeU8(0xdd)
                writeU32BE(UInt32(n))
            }
            for x in arr { pack(x) }
        case .map(let m):
            let n = m.count
            if n < 16 {
                writeU8(0x80 | UInt8(n))
            } else if n <= Int(UInt16.max) {
                writeU8(0xde)
                var be = UInt16(n).bigEndian
                Swift.withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
            } else {
                writeU8(0xdf)
                writeU32BE(UInt32(n))
            }
            for (k, val) in m {
                pack(.string(k))
                pack(val)
            }
        }
    }
}

/// Length-prefixed frame (4-byte big endian size + MessagePack body), matching `leapflow.bridge.protocol`.
enum MessageCodec {
    static func decodeFrame(_ data: Data) throws -> [String: MPValue] {
        var r = MPReader(bytes: [UInt8](data))
        let root = try r.unpack()
        guard case .map(let m) = root else { throw MessageCodecError.invalidFormat }
        return m
    }

    static func encodeFrame(_ dict: [String: MPValue]) throws -> Data {
        var w = MPWriter()
        w.pack(.map(dict))
        let body = w.data
        var out = Data()
        var len = UInt32(body.count).bigEndian
        Swift.withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(body)
        return out
    }
}

extension MPValue {
    func asString(default defaultVal: String = "") -> String {
        if case .string(let s) = self { return s }
        return defaultVal
    }

    func asInt(default defaultVal: Int64 = 0) -> Int64 {
        switch self {
        case .int(let i): return i
        case .uint(let u): return Int64(u)
        default: return defaultVal
        }
    }

    func asBool(default defaultVal: Bool = false) -> Bool {
        if case .bool(let b) = self { return b }
        return defaultVal
    }

    func asMap() -> [String: MPValue]? {
        if case .map(let m) = self { return m }
        return nil
    }

    /// Best-effort bridge to JSON-serializable structure for AX trees etc.
    func toJSONObject() -> Any {
        switch self {
        case .null:
            return NSNull()
        case .bool(let b):
            return b
        case .int(let i):
            return i
        case .uint(let u):
            return u
        case .double(let d):
            return d
        case .string(let s):
            return s
        case .binary(let d):
            return d.base64EncodedString()
        case .array(let a):
            return a.map { $0.toJSONObject() }
        case .map(let m):
            var o: [String: Any] = [:]
            for (k, v) in m { o[k] = v.toJSONObject() }
            return o
        }
    }
}

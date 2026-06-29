import ApplicationServices
import AppKit
import Foundation

enum AXTreeMonitor {
    private static let maxDepthDefault = 6
    private static let maxChildren = 40

    // MARK: - Optional AX property table
    //
    // Table-driven extra-attribute extraction. Adding a new attribute is one
    // line: append a row to the appropriate table. Each row carries the
    // "default" value used to suppress emission when the live value matches
    // (keeping payloads compact). The output key is the stable wire-name used
    // by the Python side, decoupled from the macOS AX constant.

    /// `(axAttribute, outputKey, defaultValue)`. Emit only when value != default.
    private static let boolPropTable: [(attr: CFString, key: String, defaultValue: Bool)] = [
        (kAXEnabledAttribute as CFString,  "enabled",  true),
        (kAXFocusedAttribute as CFString,  "focused",  false),
        (kAXSelectedAttribute as CFString, "selected", false),
        (kAXExpandedAttribute as CFString, "expanded", false),
    ]

    /// `(axAttribute, outputKey)`. Emit only when value is non-empty.
    private static let stringPropTable: [(attr: CFString, key: String)] = [
        (kAXPlaceholderValueAttribute as CFString, "placeholder"),
    ]

    static func readTree(bundleId: String?, maxDepth: Int = maxDepthDefault) -> MPValue {
        guard PermissionGuard.checkAccessibilityTrusted(prompt: false) else {
            return .map([
                "error": .string("accessibility_not_trusted"),
                "hint": .string("Grant Accessibility in System Settings → Privacy & Security."),
            ])
        }

        let app: NSRunningApplication?
        if let bid = bundleId, !bid.isEmpty {
            app = NSRunningApplication.runningApplications(withBundleIdentifier: bid).first
        } else {
            app = NSWorkspace.shared.frontmostApplication
        }

        guard let running = app else {
            return .map(["error": .string("app_not_found")])
        }

        let el = AXUIElementCreateApplication(running.processIdentifier)
        let root = serialize(element: el, depth: 0, maxDepth: maxDepth, pathPrefix: "")
        return .map([
            "pid": .int(Int64(running.processIdentifier)),
            "bundle_id": .string(running.bundleIdentifier ?? ""),
            "root": root,
        ])
    }

    /// Resolve a path-based node ID back to an AXUIElement.
    /// Path format: "AXWindow_0/AXToolbar_0/AXButton_2"
    static func resolveElement(bundleId: String?, nodeId: String) -> AXUIElement? {
        let app: NSRunningApplication?
        if let bid = bundleId, !bid.isEmpty {
            app = NSRunningApplication.runningApplications(withBundleIdentifier: bid).first
        } else {
            app = NSWorkspace.shared.frontmostApplication
        }
        guard let running = app else { return nil }

        let root = AXUIElementCreateApplication(running.processIdentifier)
        let segments = nodeId.split(separator: "/").map(String.init)
        guard !segments.isEmpty else { return nil }

        var current = root
        for segment in segments {
            guard let (targetRole, targetIndex) = parseSegment(segment) else { return nil }
            guard let child = findChild(of: current, role: targetRole, index: targetIndex) else {
                return nil
            }
            current = child
        }
        return current
    }

    private static func parseSegment(_ segment: String) -> (String, Int)? {
        guard let lastUnderscore = segment.lastIndex(of: "_") else { return nil }
        let role = String(segment[segment.startIndex..<lastUnderscore])
        let indexStr = String(segment[segment.index(after: lastUnderscore)...])
        guard let index = Int(indexStr) else { return nil }
        return (role, index)
    }

    private static func findChild(of element: AXUIElement, role: String, index: Int) -> AXUIElement? {
        guard let ref = copyRaw(element, kAXChildrenAttribute as CFString),
              let children = ref as? [AnyObject] else { return nil }

        var roleCount = 0
        for obj in children {
            guard CFGetTypeID(obj) == AXUIElementGetTypeID() else { continue }
            let child = unsafeBitCast(obj, to: AXUIElement.self)
            let childRole = copyAttribute(child, kAXRoleAttribute as CFString) ?? ""
            if childRole == role {
                if roleCount == index {
                    return child
                }
                roleCount += 1
            }
        }
        return nil
    }

    private static func serialize(element: AXUIElement, depth: Int, maxDepth: Int, pathPrefix: String) -> MPValue {
        if depth >= maxDepth {
            return .map(["role": .string("truncated"), "title": .string(""), "id": .string("")])
        }

        let role = copyAttribute(element, kAXRoleAttribute as CFString) ?? ""
        let title = copyAttribute(element, kAXTitleAttribute as CFString)
            ?? copyAttribute(element, kAXValueAttribute as CFString)
            ?? ""

        var childrenVals: [MPValue] = []
        guard let ref = copyRaw(element, kAXChildrenAttribute as CFString) else {
            var m: [String: MPValue] = [
                "role": .string(role),
                "title": .string(title),
                "id": .string(pathPrefix),
                "children": .array([]),
            ]
            if let frameVal = frameDict(element) { m["frame"] = frameVal }
            return .map(m)
        }

        if let arr = ref as? [AnyObject] {
            var roleCounts: [String: Int] = [:]
            for obj in arr.prefix(maxChildren) {
                guard CFGetTypeID(obj) == AXUIElementGetTypeID() else { continue }
                let ch = unsafeBitCast(obj, to: AXUIElement.self)
                let childRole = copyAttribute(ch, kAXRoleAttribute as CFString) ?? "Unknown"
                let idx = roleCounts[childRole, default: 0]
                roleCounts[childRole] = idx + 1
                let childPath = pathPrefix.isEmpty
                    ? "\(childRole)_\(idx)"
                    : "\(pathPrefix)/\(childRole)_\(idx)"
                childrenVals.append(serialize(element: ch, depth: depth + 1, maxDepth: maxDepth, pathPrefix: childPath))
            }
        }

        var m: [String: MPValue] = [
            "role": .string(role),
            "title": .string(title),
            "id": .string(pathPrefix),
            "children": .array(childrenVals),
        ]
        if let frameVal = frameDict(element) {
            m["frame"] = frameVal
        }
        let axProps = collectAxProps(element)
        if !axProps.isEmpty {
            m["ax_props"] = .map(axProps)
        }
        return .map(m)
    }

    /// Collect compact "semantic state" attributes (enabled / focused / etc.).
    /// Only non-default / non-empty values are emitted to keep the payload small.
    private static func collectAxProps(_ element: AXUIElement) -> [String: MPValue] {
        var props: [String: MPValue] = [:]
        for entry in boolPropTable {
            if let value = copyBoolAttribute(element, entry.attr), value != entry.defaultValue {
                props[entry.key] = .bool(value)
            }
        }
        for entry in stringPropTable {
            if let value = copyAttribute(element, entry.attr), !value.isEmpty {
                props[entry.key] = .string(value)
            }
        }
        return props
    }

    private static func frameDict(_ element: AXUIElement) -> MPValue? {
        var pos: CFTypeRef?
        var size: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &pos) == .success,
              AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &size) == .success,
              let p = pos, let s = size
        else { return nil }

        var point = CGPoint.zero
        var cgSize = CGSize.zero
        guard CFGetTypeID(p) == AXValueGetTypeID(),
              CFGetTypeID(s) == AXValueGetTypeID()
        else { return nil }
        let posVal = p as! AXValue
        let sizeVal = s as! AXValue
        guard AXValueGetValue(posVal, .cgPoint, &point), AXValueGetValue(sizeVal, .cgSize, &cgSize) else { return nil }

        return .map([
            "x": .double(Double(point.x)),
            "y": .double(Double(point.y)),
            "w": .double(Double(cgSize.width)),
            "h": .double(Double(cgSize.height)),
        ])
    }

    private static func copyAttribute(_ element: AXUIElement, _ attr: CFString) -> String? {
        var ref: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attr, &ref) == .success, let r = ref else { return nil }
        if let s = r as? String { return s }
        if let n = r as? NSNumber { return n.stringValue }
        return nil
    }

    /// Read a `Bool`-typed AX attribute. Returns nil if the attribute is not
    /// present or cannot be coerced. Mirrors `copyAttribute` so callers can
    /// pick the right variant per property type.
    private static func copyBoolAttribute(_ element: AXUIElement, _ attr: CFString) -> Bool? {
        var ref: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attr, &ref) == .success, let r = ref else { return nil }
        if CFGetTypeID(r) == CFBooleanGetTypeID() {
            return CFBooleanGetValue((r as! CFBoolean))
        }
        if let n = r as? NSNumber { return n.boolValue }
        return nil
    }

    private static func copyRaw(_ element: AXUIElement, _ attr: CFString) -> AnyObject? {
        var ref: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attr, &ref) == .success else { return nil }
        return ref as AnyObject?
    }
}

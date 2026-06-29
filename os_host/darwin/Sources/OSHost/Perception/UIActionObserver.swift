import ApplicationServices
import AppKit
import CoreGraphics
import Foundation

/// Observes user input events (mouse, keyboard, scroll) via CGEvent tap
/// and broadcasts normalized UI action events to connected clients.
///
/// Implements throttling to avoid event storms:
/// - Keyboard: 50ms coalesce window
/// - Scroll: 200ms coalesce window
final class UIActionObserver: UIActionProvider {

    // MARK: - Configuration

    private enum Throttle {
        static let keyboardMs: TimeInterval = 0.050
        static let scrollMs: TimeInterval = 0.200
        static let dragThresholdPx: CGFloat = 5.0
    }

    // MARK: - Dependencies

    private let broadcaster: ClientBroadcaster
    private weak var recordingMode: RecordingMode?

    // MARK: - Drag State Machine

    private enum DragState {
        case idle
        case potential(startPoint: CGPoint, startTime: TimeInterval, startElement: (role: String, label: String, identifier: String), app: String)
        case dragging(startPoint: CGPoint, startElement: (role: String, label: String, identifier: String), app: String)
    }

    // MARK: - State

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var observerThread: Thread?
    private let lock = NSLock()

    private var _isObserving = false
    private var lastKeyboardPush: TimeInterval = 0
    private var lastScrollPush: TimeInterval = 0
    private var pendingKeyEvent: [String: MPValue]?
    private var pendingScrollEvent: [String: MPValue]?
    private var keyCoalesceTimer: DispatchWorkItem?
    private var scrollCoalesceTimer: DispatchWorkItem?
    private var dragState: DragState = .idle

    private let workQueue = DispatchQueue(label: "com.leapflow.ui-action-observer", qos: .userInteractive)

    // MARK: - Recording-aware throttle

    private var keyboardThrottle: TimeInterval {
        (recordingMode?.isActive == true) ? 0.020 : Throttle.keyboardMs
    }

    private var scrollThrottle: TimeInterval {
        (recordingMode?.isActive == true) ? 0.080 : Throttle.scrollMs
    }

    // MARK: - UIActionProvider

    var isObserving: Bool {
        lock.lock(); defer { lock.unlock() }
        return _isObserving
    }

    init(broadcaster: ClientBroadcaster) {
        self.broadcaster = broadcaster
    }

    func configure(recordingMode: RecordingMode) {
        self.recordingMode = recordingMode
    }

    @discardableResult
    func startObserving() -> Bool {
        lock.lock()
        guard !_isObserving else { lock.unlock(); return true }
        lock.unlock()

        guard PermissionGuard.checkAccessibilityTrusted(prompt: false) else {
            fputs("UIActionObserver: accessibility not trusted, cannot start\n", stderr)
            return false
        }

        let eventsOfInterest: CGEventMask =
            (1 << CGEventType.leftMouseDown.rawValue) |
            (1 << CGEventType.leftMouseUp.rawValue) |
            (1 << CGEventType.leftMouseDragged.rawValue) |
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.scrollWheel.rawValue)

        let refcon = Unmanaged.passUnretained(self).toOpaque()

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: eventsOfInterest,
            callback: UIActionObserver.eventTapCallback,
            userInfo: refcon
        ) else {
            fputs("UIActionObserver: failed to create CGEvent tap\n", stderr)
            return false
        }

        guard let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0) else {
            fputs("UIActionObserver: failed to create run loop source\n", stderr)
            return false
        }

        self.eventTap = tap
        self.runLoopSource = source

        // Run event tap on a dedicated thread to avoid blocking main
        let thread = Thread {
            let rl = CFRunLoopGetCurrent()
            CFRunLoopAddSource(rl, source, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
            CFRunLoopRun()
        }
        thread.name = "UIActionObserver-EventTap"
        thread.qualityOfService = .userInteractive
        thread.start()
        self.observerThread = thread

        lock.lock()
        _isObserving = true
        lock.unlock()

        fputs("UIActionObserver: started observing user input events\n", stderr)
        return true
    }

    func stopObserving() {
        lock.lock()
        guard _isObserving else { lock.unlock(); return }
        _isObserving = false
        keyCoalesceTimer?.cancel()
        scrollCoalesceTimer?.cancel()
        keyCoalesceTimer = nil
        scrollCoalesceTimer = nil
        pendingKeyEvent = nil
        pendingScrollEvent = nil
        dragState = .idle
        lock.unlock()

        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let source = runLoopSource, let thread = observerThread {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
            thread.cancel()
        }
        eventTap = nil
        runLoopSource = nil
        observerThread = nil

        fputs("UIActionObserver: stopped\n", stderr)
    }

    deinit {
        stopObserving()
    }

    // MARK: - CGEvent Tap Callback

    private static let eventTapCallback: CGEventTapCallBack = { _, type, event, refcon in
        guard let refcon else { return Unmanaged.passUnretained(event) }
        let observer = Unmanaged<UIActionObserver>.fromOpaque(refcon).takeUnretainedValue()

        switch type {
        case .leftMouseDown:
            observer.handleMouseDown(event)
        case .leftMouseDragged:
            observer.handleMouseDragged(event)
        case .leftMouseUp:
            observer.handleMouseUp(event)
        case .keyDown:
            observer.handleKeyDown(event)
        case .scrollWheel:
            observer.handleScroll(event)
        case .tapDisabledByTimeout, .tapDisabledByUserInput:
            if let tap = observer.eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
        default:
            break
        }
        return Unmanaged.passUnretained(event)
    }

    // MARK: - Mouse / Drag FSM

    private func handleMouseDown(_ event: CGEvent) {
        let location = event.location
        let ts = Date().timeIntervalSince1970
        let (role, label, identifier) = hitTestElement(at: location)
        let bundleId = frontmostBundleId()

        lock.lock()
        dragState = .potential(
            startPoint: location,
            startTime: ts,
            startElement: (role, label, identifier),
            app: bundleId
        )
        lock.unlock()
    }

    private func handleMouseDragged(_ event: CGEvent) {
        lock.lock()
        switch dragState {
        case .potential(let start, _, let el, let app):
            let current = event.location
            let dx = current.x - start.x
            let dy = current.y - start.y
            if (dx * dx + dy * dy) > Throttle.dragThresholdPx * Throttle.dragThresholdPx {
                dragState = .dragging(startPoint: start, startElement: el, app: app)
            }
        default:
            break
        }
        lock.unlock()
    }

    private func handleMouseUp(_ event: CGEvent) {
        let endLocation = event.location
        let ts = Date().timeIntervalSince1970

        lock.lock()
        let state = dragState
        dragState = .idle
        lock.unlock()

        switch state {
        case .dragging(let startPoint, let startElement, let startApp):
            let (endRole, endLabel, endId) = hitTestElement(at: endLocation)
            let endApp = frontmostBundleId()
            let payload: [String: MPValue] = [
                "action": .string("drag_complete"),
                "start_x": .double(Double(startPoint.x)),
                "start_y": .double(Double(startPoint.y)),
                "end_x": .double(Double(endLocation.x)),
                "end_y": .double(Double(endLocation.y)),
                "start_element_role": .string(startElement.role),
                "start_element_label": .string(startElement.label),
                "start_element_id": .string(startElement.identifier),
                "end_element_role": .string(endRole),
                "end_element_label": .string(endLabel),
                "end_element_id": .string(endId),
                "start_app": .string(startApp),
                "end_app": .string(endApp),
                "cross_app": .bool(startApp != endApp),
                "app_bundle_id": .string(endApp),
                "timestamp": .double(ts),
            ]
            pushUIAction(payload)

        case .potential(_, _, let el, let app):
            let payload: [String: MPValue] = [
                "action": .string("click"),
                "element_role": .string(el.role),
                "element_label": .string(el.label),
                "element_id": .string(el.identifier),
                "mouse_x": .double(Double(endLocation.x)),
                "mouse_y": .double(Double(endLocation.y)),
                "app_bundle_id": .string(app),
                "timestamp": .double(ts),
            ]
            pushUIAction(payload)

        case .idle:
            break
        }
    }

    // MARK: - Keyboard

    private func handleKeyDown(_ event: CGEvent) {
        let ts = Date().timeIntervalSince1970
        let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
        let flags = event.flags

        let modifiers = extractModifiers(flags)
        let isShortcut = !modifiers.isEmpty
        let action = isShortcut ? "shortcut" : "type"
        let bundleId = frontmostBundleId()

        // Extract unicode character (keyboardGetUnicodeString is inline, no IPC)
        var char = ""
        if !isShortcut {
            var actualLen = 0
            var unicodeChars = [UniChar](repeating: 0, count: 4)
            event.keyboardGetUnicodeString(
                maxStringLength: 4,
                actualStringLength: &actualLen,
                unicodeString: &unicodeChars
            )
            if actualLen > 0 {
                char = String(utf16CodeUnits: unicodeChars, count: actualLen)
            }
        }

        var payload: [String: MPValue] = [
            "action": .string(action),
            "key_code": .int(Int64(keyCode)),
            "modifiers": .array(modifiers.map { .string($0) }),
            "app_bundle_id": .string(bundleId),
            "timestamp": .double(ts),
        ]
        if !char.isEmpty {
            payload["char"] = .string(char)
        }

        if isShortcut {
            pushUIAction(payload)
            return
        }

        throttleKeyboard(payload: payload, ts: ts)
    }

    // MARK: - Scroll

    private func handleScroll(_ event: CGEvent) {
        let ts = Date().timeIntervalSince1970
        let deltaY = event.getIntegerValueField(.scrollWheelEventDeltaAxis1)
        let deltaX = event.getIntegerValueField(.scrollWheelEventDeltaAxis2)
        let location = event.location
        let bundleId = frontmostBundleId()

        let payload: [String: MPValue] = [
            "action": .string("scroll"),
            "mouse_x": .double(Double(location.x)),
            "mouse_y": .double(Double(location.y)),
            "delta_x": .int(Int64(deltaX)),
            "delta_y": .int(Int64(deltaY)),
            "app_bundle_id": .string(bundleId),
            "timestamp": .double(ts),
        ]

        throttleScroll(payload: payload, ts: ts)
    }

    // MARK: - Throttling

    private func throttleKeyboard(payload: [String: MPValue], ts: TimeInterval) {
        workQueue.async { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.keyCoalesceTimer?.cancel()
            self.pendingKeyEvent = payload
            self.lock.unlock()

            let item = DispatchWorkItem { [weak self] in
                guard let self else { return }
                self.lock.lock()
                guard var pending = self.pendingKeyEvent else {
                    self.lock.unlock()
                    return
                }
                self.pendingKeyEvent = nil
                self.lastKeyboardPush = Date().timeIntervalSince1970
                self.lock.unlock()

                // Strip char from secure fields (AX query runs here, off event-tap thread)
                if pending["char"] != nil && self.isSecureField() {
                    pending.removeValue(forKey: "char")
                }
                self.pushUIAction(pending)
            }
            self.lock.lock()
            self.keyCoalesceTimer = item
            self.lock.unlock()
            self.workQueue.asyncAfter(deadline: .now() + self.keyboardThrottle, execute: item)
        }
    }

    private func throttleScroll(payload: [String: MPValue], ts: TimeInterval) {
        workQueue.async { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.scrollCoalesceTimer?.cancel()
            self.pendingScrollEvent = payload
            self.lock.unlock()

            let item = DispatchWorkItem { [weak self] in
                guard let self else { return }
                self.lock.lock()
                guard let pending = self.pendingScrollEvent else {
                    self.lock.unlock()
                    return
                }
                self.pendingScrollEvent = nil
                self.lastScrollPush = Date().timeIntervalSince1970
                self.lock.unlock()
                self.pushUIAction(pending)
            }
            self.lock.lock()
            self.scrollCoalesceTimer = item
            self.lock.unlock()
            self.workQueue.asyncAfter(deadline: .now() + self.scrollThrottle, execute: item)
        }
    }

    // MARK: - Event Broadcasting

    private func pushUIAction(_ payload: [String: MPValue]) {
        let frame: [String: MPValue] = [
            "v": .int(1),
            "type": .string("event"),
            "event": .string("event.ui_action"),
            "payload": .map(payload),
            "ts": .double(ProcessInfo.processInfo.systemUptime),
        ]
        guard let data = try? MessageCodec.encodeFrame(frame) else { return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.broadcaster.broadcast(data: data)
        }
    }

    // MARK: - AX Hit-Test

    private func hitTestElement(at point: CGPoint) -> (role: String, label: String, identifier: String) {
        guard let app = NSWorkspace.shared.frontmostApplication else {
            return ("", "", "")
        }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        var element: AXUIElement?
        let err = AXUIElementCopyElementAtPosition(axApp, Float(point.x), Float(point.y), &element)
        guard err == .success, let el = element else {
            return ("", "", "")
        }

        let role = axStringAttribute(el, kAXRoleAttribute as CFString) ?? ""
        let label = axStringAttribute(el, kAXDescriptionAttribute as CFString)
            ?? axStringAttribute(el, kAXTitleAttribute as CFString)
            ?? axStringAttribute(el, kAXValueAttribute as CFString)
            ?? ""
        let identifier = axStringAttribute(el, kAXIdentifierAttribute as CFString) ?? ""

        return (role, label, identifier)
    }

    private func axStringAttribute(_ element: AXUIElement, _ attr: CFString) -> String? {
        var ref: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attr, &ref) == .success, let r = ref else { return nil }
        if let s = r as? String { return s }
        return nil
    }

    // MARK: - Helpers

    private func frontmostBundleId() -> String {
        NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? ""
    }

    /// Detect if the currently focused element is a secure text field (password).
    private func isSecureField() -> Bool {
        guard let app = NSWorkspace.shared.frontmostApplication else { return false }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(axApp, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let el = focused else { return false }
        let element = el as! AXUIElement  // CFTypeRef from AX API is always AXUIElement
        let role = axStringAttribute(element, kAXRoleAttribute as CFString)
        return role == "AXSecureTextField"
    }

    private func extractModifiers(_ flags: CGEventFlags) -> [String] {
        var mods: [String] = []
        if flags.contains(.maskCommand) { mods.append("command") }
        if flags.contains(.maskControl) { mods.append("control") }
        if flags.contains(.maskAlternate) { mods.append("option") }
        if flags.contains(.maskShift) { mods.append("shift") }
        return mods
    }
}

import CoreGraphics
import Darwin
import Dispatch
import Foundation

// MARK: - CLI Argument Parsing

struct LaunchOptions {
    var socketPath: String?
    var pidFile: String?
    var logFile: String?
    var daemon: Bool = false
    var promptAX: Bool = false
}

func parseArguments(_ argv: [String]) -> LaunchOptions {
    var opts = LaunchOptions()
    var i = 1
    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--socket":
            if i + 1 < argv.count {
                opts.socketPath = argv[i + 1]
                i += 2
            } else {
                fputs("OSHost: --socket requires a value\n", stderr)
                i += 1
            }
        case "--pid-file":
            if i + 1 < argv.count {
                opts.pidFile = argv[i + 1]
                i += 2
            } else {
                fputs("OSHost: --pid-file requires a value\n", stderr)
                i += 1
            }
        case "--log-file":
            if i + 1 < argv.count {
                opts.logFile = argv[i + 1]
                i += 2
            } else {
                fputs("OSHost: --log-file requires a value\n", stderr)
                i += 1
            }
        case "--daemon":
            opts.daemon = true
            i += 1
        case "--prompt-ax":
            opts.promptAX = true
            i += 1
        default:
            i += 1
        }
    }
    return opts
}

let options = parseArguments(ProcessInfo.processInfo.arguments)

// MARK: - Log file redirection (must happen before any stderr writes we want captured)

if let logPath = options.logFile {
    let expanded = (logPath as NSString).expandingTildeInPath
    if freopen(expanded, "a", stderr) == nil {
        // Can't log to file; fall back to original stderr.
        perror("OSHost: freopen(log-file)")
    } else {
        setvbuf(stderr, nil, _IOLBF, 0)
    }
}

// MARK: - Resolve socket path: --socket > env > default

let socketPath: String = {
    if let cli = options.socketPath { return cli }
    if let env = ProcessInfo.processInfo.environment["LEAPFLOW_BRIDGE_SOCKET"] { return env }
    return "/tmp/leapflow.sock"
}()

// MARK: - PID file

func writePidFile(_ path: String) {
    let expanded = (path as NSString).expandingTildeInPath
    let pid = getpid()
    let content = "\(pid)\n"
    do {
        try content.write(toFile: expanded, atomically: true, encoding: .utf8)
    } catch {
        fputs("OSHost: failed to write pid-file \(expanded): \(error)\n", stderr)
    }
}

if let pidFile = options.pidFile {
    writePidFile(pidFile)
}

// MARK: - Cleanup

let cleanupOnce = DispatchQueue(label: "oshost.cleanup")
var cleanupDone = false

func performCleanup() {
    cleanupOnce.sync {
        if cleanupDone { return }
        cleanupDone = true
        if let pidFile = options.pidFile {
            let expanded = (pidFile as NSString).expandingTildeInPath
            unlink(expanded)
        }
        let sockExpanded = (socketPath as NSString).expandingTildeInPath
        unlink(sockExpanded)
    }
}

// MARK: - Signal handling via DispatchSource

// Module-scope storage so the DispatchSourceSignal objects are not deallocated.
var _signalSources: [DispatchSourceSignal] = []

func installSignalHandlers() {
    // Ignore default disposition so DispatchSource can observe.
    signal(SIGTERM, SIG_IGN)
    signal(SIGINT, SIG_IGN)
    signal(SIGHUP, SIG_IGN)
    signal(SIGPIPE, SIG_IGN)

    let queue = DispatchQueue(label: "oshost.signals")

    let term = DispatchSource.makeSignalSource(signal: SIGTERM, queue: queue)
    term.setEventHandler {
        fputs("OSHost: SIGTERM received, shutting down\n", stderr)
        performCleanup()
        exit(0)
    }
    term.resume()

    let intr = DispatchSource.makeSignalSource(signal: SIGINT, queue: queue)
    intr.setEventHandler {
        fputs("OSHost: SIGINT received, shutting down\n", stderr)
        performCleanup()
        exit(0)
    }
    intr.resume()

    let hup = DispatchSource.makeSignalSource(signal: SIGHUP, queue: queue)
    hup.setEventHandler {
        fputs("OSHost: SIGHUP received (reload not yet implemented)\n", stderr)
    }
    hup.resume()

    // Keep references alive for the lifetime of the process.
    _signalSources = [term, intr, hup]
}

installSignalHandlers()

atexit {
    performCleanup()
}

// MARK: - Permission checks

let shouldPrompt = options.promptAX || !options.daemon

if !PermissionGuard.checkAccessibilityTrusted(prompt: false) {
    if shouldPrompt {
        fputs("OSHost: Accessibility not granted — requesting permission...\n", stderr)
        _ = PermissionGuard.checkAccessibilityTrusted(prompt: true)
    } else {
        fputs("OSHost: Accessibility not granted — UI events will be unavailable\n", stderr)
    }
}

if !CGPreflightScreenCaptureAccess() {
    if shouldPrompt {
        fputs("OSHost: Screen Recording not granted — requesting permission...\n", stderr)
        CGRequestScreenCaptureAccess()
    } else {
        fputs("OSHost: Screen Recording not granted — screen capture will be unavailable\n", stderr)
    }
}

// MARK: - Component initialisation

let broadcaster = ClientBroadcaster()
let router = RpcRouter()
router.broadcaster = broadcaster
router.startClipboardPolling()

let appFocusMonitor = AppFocusMonitor(broadcaster: broadcaster)
appFocusMonitor.start()
router.appFocusMonitor = appFocusMonitor

if let screenProvider = ProviderFactory.makeScreenCaptureProvider(broadcaster: broadcaster) {
    router.screenCaptureProvider = screenProvider
    if screenProvider.permissionGranted {
        router.capabilities.capabilities.insert("screen.capture")
        router.capabilities.capabilities.insert("video.recording")
        fputs("OSHost: ScreenCaptureMonitor active\n", stderr)
    } else {
        fputs("OSHost: ScreenCaptureMonitor created but Screen Recording permission NOT granted\n", stderr)
        fputs("OSHost: → Grant permission in System Settings → Privacy & Security → Screen Recording\n", stderr)
    }
} else {
    router.capabilities.capabilities.remove("screen.capture")
    fputs("OSHost: ScreenCaptureMonitor unavailable (macOS 14+ required)\n", stderr)
}

let uiActionProvider = ProviderFactory.makeUIActionProvider(broadcaster: broadcaster)
if let observer = uiActionProvider as? UIActionObserver {
    observer.configure(recordingMode: router.recordingMode)
}
if uiActionProvider.startObserving() {
    fputs("OSHost: UIActionObserver active\n", stderr)
} else {
    fputs("OSHost: UIActionObserver cannot start — Accessibility permission NOT granted\n", stderr)
    fputs("OSHost: → Grant permission in System Settings → Privacy & Security → Accessibility\n", stderr)
}

fputs("OSHost platform: \(router.capabilities.platformId) (\(router.capabilities.osVersion))\n", stderr)
fputs("OSHost capabilities: \(router.capabilities.capabilities.sorted().joined(separator: ", "))\n", stderr)

if !options.daemon {
    // Interactive convenience output for terminal users; suppressed in daemon mode.
    print("OSHost ready on \(socketPath)")
    print("OSHost capabilities: \(router.capabilities.capabilities.sorted().joined(separator: ", "))")
}

SocketServer.serve(socketPath: socketPath, router: router, broadcaster: broadcaster)

performCleanup()

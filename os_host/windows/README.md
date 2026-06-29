# LeapFlow OS Host — Windows

Platform-specific host implementation for Windows 10/11.

## Status

**Not yet implemented.** This is a placeholder for future development.

## Planned Technology

- **Language:** Rust
- **IPC:** Named Pipes (or TCP localhost fallback)
- **Perception:** UIAutomation, ReadDirectoryChangesW, Windows.Graphics.Capture
- **Execution:** Win32 API, PowerShell remoting, SendInput
- **Security:** UAC integration, Windows Security Center

## Protocol Conformance

Must implement all methods defined in `../protocol/rpc_schema.yaml`.
Capabilities declared at handshake will vary from the darwin implementation.

## Build

```bash
# (future) from repo root:
make host-build   # auto-detects platform
```

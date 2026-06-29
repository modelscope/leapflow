# LeapFlow OS Host — Linux

Platform-specific host implementation for Linux (X11/Wayland).

## Status

**Not yet implemented.** This is a placeholder for future development.

## Planned Technology

- **Language:** Rust
- **IPC:** Unix domain socket (same framing as darwin)
- **Perception:** AT-SPI (accessibility), inotify (fs watch), PipeWire/X11 (screen capture)
- **Execution:** D-Bus, xdotool/ydotool, subprocess
- **Security:** polkit, AppArmor/SELinux integration

## Protocol Conformance

Must implement all methods defined in `../protocol/rpc_schema.yaml`.
Capabilities declared at handshake will vary from the darwin implementation.

## Build

```bash
# (future) from repo root:
make host-build   # auto-detects platform
```

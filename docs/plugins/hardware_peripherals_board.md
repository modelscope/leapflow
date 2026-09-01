# Peripherals on LeapBoard: Discovery, Preview, and Settings

> **Audience**: Driver and app-pack developers adding a peripheral, and operators
> deciding what a profile should expose.
> **Authoritative source**: Derived from production code at
> `src/leapflow/hardware/providers/` (discovery), `src/leapflow/hardware/transports/`
> (capture and control), `src/leapflow/hardware/context.py` (declared facts),
> `src/leapflow/hardware/preview.py` (preview lease),
> `src/leapflow/hardware/risk.py` (privacy classification), and
> `src/leapflow/dashboard/` (board data plane and view).
> **Scope**: how a peripheral becomes visible, previewable and settable on LeapBoard,
> and what a third party must implement to add one. It does **not** describe device
> readiness or calibration procedures — see
> [`hardware_init_calibration.md`](hardware_init_calibration.md).

---

## 1. The claim this document makes

**Adding a peripheral requires no board code.** A new device appears on LeapBoard with
live values, a trace, a preview and controls because every panel is derived from the
*declaration* — not from a per-device view, an icon table, or a type switch.

That is a contract, and it is testable. The board asks two questions of each channel,
both answered by declared fields:

| Declared | Board renders |
|---|---|
| `representation: scalar` + `sample_rate_hz > 0` | value, trend, sparkline, envelope band |
| `representation: state` | value as-is |
| `representation: frame` | **preview panel** (`MediaPreview`) |
| `direction: readwrite` / `write` | **control** whose widget comes from the envelope |
| `privacy: environment` / `personal` | consent notice, and the read is gated |

Nothing consults `device_class`. It is a free-form grouping label used for section
headings and nothing else, deliberately not an enum: the moment a device *type* decides
what is permitted, every new peripheral needs a core edit and an unrecognised one gets a
wrong default.

---

## 2. Adding a peripheral: five steps

### 2.1 Implement a `HardwareContextProvider`

```python
class MyScannerProvider:
    kind = "my_scanner"

    def discover(self) -> tuple[HardwareContext, ...]:
        ...
```

`discover()` **must not connect to a device**. Discovery has to work with the hardware
powered off, and it runs during daemon boot — so it must also not block. Two rules follow
that are easy to get wrong:

- **No device I/O, and no slow subprocess.** Reading a mount table, a sysfs node or an
  in-process counter is fine. Opening a camera is not: on macOS that raises a system
  permission dialog, and a background process cannot explain why one appeared.
- **Enumerate metadata only.** `leapflow.hardware.media` lists AVFoundation inputs by
  parsing ffmpeg's own `-list_devices` output precisely because it never opens one.

### 2.2 Implement a `HardwareTransport`

Six methods: `open`, `close`, `read`, `write`, `probe`, `halt`. See
`src/leapflow/hardware/transport.py`.

If the device produces images, additionally satisfy `FrameTransport`:

```python
async def read_frame(
    self, channel_id: str, *, max_width: int = 0, quality: int = 0
) -> FrameReading: ...
```

This is a **side protocol, not a seventh core method**. Capability is discovered with
`isinstance(transport, FrameTransport)`, so most drivers never grow a method they cannot
implement. A device declaring a `frame` channel whose transport does not satisfy it is
refused on first preview with `failure_code="transport_not_frame_capable"` — a named
degradation rather than an `AttributeError`.

`FrameReading` is deliberately **not** a `Reading`. Readings are appended to raw NDJSON
segments and downsampled into DuckDB windows; a frame has no mean and no bound, and a few
hundred kilobytes per sample would turn the segment writer into a disk filler with a
schedule.

### 2.3 Register both

Three ways, in ascending order of independence:

```python
# In-tree: one row in the factory table.
_PROVIDERS["my_scanner"] = "my_pkg.provider:build_provider"

# From a plugin, scoped so a hot reload cannot leave a stale factory behind.
scope.effect(register_provider("my_scanner", "my_pkg.provider:build_provider"))
scope.effect(register_transport("my_rig", "my_pkg.driver:build_transport"))

# Out-of-tree: `pip install` is enough.
[project.entry-points."leapflow.hardware.providers"]
my_scanner = "my_pkg.provider:build_provider"
[project.entry-points."leapflow.hardware.transports"]
my_rig = "my_pkg.driver:build_transport"
```

Built-in names win over entry points: an installed package must not be able to hijack
`yaml`, `host` or `media` and change where a profile's device knowledge comes from.

### 2.4 Declare the channels

```yaml
channels:
  - channel_id: frame
    direction: read
    quantity: image_frame
    representation: frame        # -> preview panel
    media_type: image/jpeg
    privacy: environment         # -> consent gate
    sample_rate_hz: 2.0          # capture *ceiling*, not a sampling cadence
  - channel_id: exposure_us
    direction: readwrite
    effect: configure            # -> hw_configure owns it
    unit: us
    envelope:
      declared: true             # -> slider, min..max, step from quantization
      min_value: 100.0
      max_value: 33000.0
      quantization: 100.0
      reversible: true
```

Two field semantics are load-bearing and non-obvious:

- **`sample_rate_hz` on a media channel is a capture ceiling.** `Channel.is_streaming`
  is false for media, so no sampling loop is built and nothing is written to the reading
  store. The preview path reads it as the fastest it may ask the device for frames.
- **The envelope *is* the widget specification.** `allowed_values` → select;
  `min_value` + `max_value` → slider stepped by `quantization`; neither → plain field.

### 2.5 Nothing else

`leap hw scan` (or the daemon's rediscovery interval) admits it, and it appears on the
fleet board grouped by `device_class`, with a device page carrying whatever its channels
declared.

---

## 3. Privacy: why a read can need consent

`HardwareEffect` classifies what a **write** changes. It cannot express the difference
between a thermometer and a webcam: both are `effect: read`, with no envelope and nothing
to actuate, and one of them discloses the room.

`PrivacyTier` is the declared fact that can:

| Tier | Meaning | Read is |
|---|---|---|
| `none` (default) | discloses nothing about the surroundings | free |
| `environment` | observes the space around the machine (camera, microphone) | gated, MEDIUM |
| `personal` | observes the person using it (screen, location) | gated, HIGH |

Consequences, all enforced in code:

- **`allow_permanent=False`.** A standing, unexpirable grant to observe somebody's room
  is not consent — it is the absence of it. A session-scoped grant still spares them a
  prompt per frame.
- **Refusal precedes the transport.** `HardwareTools._consent_for_read` returns before
  `registry.transport()` is called, because opening the device is what raises the
  platform dialog. A refused read must never get that far.
- **Fail closed, three ways.** No gate installed (the in-process CLI binds none), a gate
  that raises, and a gate that denies all produce `failure_code="consent_required"` with
  a message naming the next step.

### 3.1 Where consent is actually given

A browser cannot grant itself a camera *by asserting so* — but it can carry the question
and the answer. Both surfaces work, and they differ only in where the prompt appears:

| Surface | Prompt appears | Use when |
|---|---|---|
| **The board page** | inline, inside the preview panel that made the request | you clicked *Start preview* and are looking at it |
| **`/board preview <id>`** | in the TUI | you want the grant before opening a browser |

Both reach the same gate. What makes the page a legitimate surface is structural, not a
relaxation:

- The prompt is **raised by the daemon's approval chain**, not invented in JavaScript. It
  arrives carrying the risk assessment and *the choices the policy allowed* — the page
  renders those verbatim, so it cannot offer an "always allow" the policy withheld.
- The answer goes back through **`approval.resolve`**, so the grant, the audit record and
  the decision semantics stay the orchestrator's.
- The prompt only exists because **this page made the request**. The person answering is
  the person who clicked.

The mechanism is `_APPROVAL_ROUTED_METHODS` in `daemon/server.py`.
`ApprovalCoordinator.request_approval` returns `deny` when no approval route is installed,
and the daemon installs one for `command.execute` and for the two device observations
(`hardware.frame`, `hardware.read`). A routed request delivers its prompt as an interleaved
`stream.chunk` notification on its own socket, which
`DaemonClient.request(on_stream_event=...)` forwards — the dashboard forwards it to the
browser hub, and the request **waits** for the answer.

That waiting is the point: answering completes the very request that raised the prompt, so
there is no second round trip and no window where a grant exists but the picture does not.
An unanswered prompt cannot leak, because a routed request registers, denies-on-exit and
unregisters exactly as a slash command does.

A local environment camera uses a **session consent family**, not a per-device prompt:
one consent covers its probe, the following MJPEG stream, and another local camera looking
at the same physical space. Camera and microphone remain separate families; personal,
remote and unknown device classes remain per-device. The action summary, risk assessment
and audit record still name the actual device/channel, so only reusable grant identity is
grouped.

The page promotes **Allow for this session** as the primary choice. **Allow once** returns
exactly one still frame or one level sample; it deliberately does not open a continuous
stream, so it never turns a one-shot decision into ongoing observation.

### 3.2 Screens

Screen-capture devices are **not enumerated by default** (`hardware.media_screens`). A
platform that presents the display as just another video input would otherwise put
"stream this person's screen" on the board beside the webcam, one click away.

---

## 4. Operating it from the TUI

`/board` is the operator surface, and every verb below reads or requests — none of them
commands a device directly.

| Command | Does |
|---|---|
| `/board hardware` | the fleet: every attached peripheral, grouped by class |
| `/board devices` | the same list as text in the TUI — no browser needed |
| `/board device <id>` | the **same** `hardware` lens, focused on one device |
| `/board preview <id> [channel]` | establish consent (prompt appears here), then open the preview |
| `/board rescan` | re-run discovery after a hot-plug |

There is one hardware lens, not two. `hardware` renders the fleet; naming a device renders
that device. They were separate templates and the split did not pay for itself —
`hardware` and `hardware_device` read as synonyms in the lens list, and the second was
never a different *way of looking*, only a different subject.

`<id>` accepts a **unique prefix**, matching how `/board stop` already resolves a watch
id: discovered ids are long (`camera_0_macbook_pro`) and an ambiguous prefix reports the
candidates rather than guessing. Deliberately no completion is offered for the id — it
would be captured when the TUI started and keep offering a device that has since been
unplugged.

`/board rescan` is ungated because every provider in the default set enumerates
passively. A scanner that transmits or leaves the host would need its own gate, which is
exactly why those are not in the default set.

---

## 5. Preview: shared, bounded, self-releasing

Two shapes, one gate. A **frame** channel streams pictures; a privacy-gated **scalar**
channel is a live meter, which is how a microphone's input level is presented. Both are
continuous disclosures of the surroundings, both need consent, and both are useless as a
static table cell. `inventory._is_previewable` decides from the declared representation, so
a non-camera frame source and a non-microphone level source work without a new case.

| Channel | Endpoint | Transport |
|---|---|---|
| `representation: frame` | `GET /api/media/stream` | MJPEG (`multipart/x-mixed-replace`) |
| privacy-gated scalar | `GET /api/media/level` | JSON, polled four times a second into a browser-local waveform |

MJPEG because an `<img>` renders it natively — no player, no codec, no decode path in
JavaScript. The level channel is polled instead: the value is a number, so there is no
response to hold open. The board draws its last 96 readings as a **browser-local waveform**;
values are never persisted as audio history.

The client asks for one frame (or one reading) first. That probe is what surfaces a
refusal as text — an `<img>` cannot report *why* it failed, because `onerror` carries no
body. If the person chooses **Allow once**, that probe is the complete result; choosing
**Allow for this session** starts the continuous stream without asking a second time.

`PreviewBroker` (`src/leapflow/hardware/preview.py`) owns the one path where a device
stays claimed across requests. Three properties, each present because its absence is a
real failure:

1. **Shared upstream.** Most devices admit a single reader, so two viewers of one camera
   must not open it twice. Frames are captured once per channel and handed to whoever
   asks.
2. **Profile-bounded work.** The page offers Economy (640px / 4fps / JPEG 60), Balanced
   (960px / 8fps / JPEG 75, default) and Detail (1280px / 12fps / JPEG 85). The daemon
   clamps every request against the channel declaration and `hardware.preview_*` ceilings;
   a hand-edited URL cannot raise capture cost. Profile identity is part of the cached
   frame key, so selecting Detail never shows a cached Economy frame.
3. **Self-releasing.** A browser tab closing is not an event the daemon can observe, so
   the lease expires on **silence**: no frame requested within
   `hardware.preview_idle_timeout_s` drops the transport, which is what actually powers
   the camera down.

The Preview selector is **not** a durable config editor. It records a browser-local
preference per device/channel and sends a bounded request for the next live preview; the
daemon is authoritative for every compute limit. The default balanced profile is chosen
to make a camera useful without spending Detail's CPU/bandwidth in every open board.

The wire path is `MediaPreview` → `GET /api/media/stream` (MJPEG) → `hardware.frame` RPC
→ broker → `read_frame`.

---

## 6. Settings: the board asks, it never decides

A control on a device page has two buttons, and the pair is the design:

| Button | RPC | Effect |
|---|---|---|
| **Preview change** | `hardware.write_request` with `dry_run=true` | every feasibility check runs; nothing reaches the device |
| **Request approval** | `hardware.write_request` with `dry_run=false` | goes through `ApprovalOrchestrator` per invocation |

Both reach the **same** daemon RPC, which delegates to the ordinary
`hw_configure`/`hw_actuate`/`hw_dispense` handler. There is deliberately no second write
path: the handler owns every feasibility check, the approval descriptor, the audit record
and the side-effect verdict, and a parallel implementation would be a second gate free to
disagree with the one that actually protects the device.

The tool is chosen from the channel's **declared effect class**, never from the caller.
Letting the board name it would allow routing a motion command on an `actuate` channel
through `hw_configure` and getting the gentler classification.

The board may carry an approval prompt raised by its own preview request and resolve it
through `approval.resolve`; the daemon still owns risk classification, policy, grants and
audit. The board cannot invent choices or approve an unrelated action. Device controls use
the same `hardware.write_request` path described above.

---

## 7. Configuration

| Key | Default | Notes |
|---|---|---|
| `hardware.enabled` | `true` | passive host/media discovery is on; reads that disclose surroundings still need consent |
| `hardware.providers` | `yaml,host,media` | comma-separated; scanners that transmit or leave the host are opt-in |
| `hardware.host_interval_s` | `5.0` | fast host channels; disk/battery/thermal multiply it |
| `hardware.host_include` / `_exclude` | empty | channel-id **prefixes**, because mounts and interfaces are discovered |
| `hardware.media_screens` | `false` | enumerate displays as previewable devices |
| `hardware.media_microphones` | `true` | level only, never audio |
| `hardware.preview_max_fps` | `12.0` | hard ceiling; the page defaults to Balanced at 8fps |
| `hardware.preview_max_width` | `1280` | hard ceiling; Balanced requests 960px, height follows aspect ratio |
| `hardware.preview_quality` | `85` | hard JPEG-quality ceiling; Balanced requests 75 |
| `hardware.preview_idle_timeout_s` | `15.0` | silence after which the device is released |
| `hardware.rediscover_interval_s` | `0` (off) | automatic rediscovery; runs on the monitor cadence, never on a turn |

All of `hardware.*` is restart-required: providers run at startup and the preview broker
is built with these values.

---

## 8. What the host provider exposes

The machine LeapFlow runs on is a device whose channel set is discovered rather than
written down. It is **one** device (`host`) with namespaced channels — `cpu.utilization`,
`memory.available_bytes`, `disk.<mount>.free_bytes`, `net.<iface>.rx_bytes_per_s`,
`battery.percent`, `thermal.<sensor>.celsius` — because seven devices would consume most
of `hardware.max_devices` before a single real peripheral was admitted.

`psutil` is an optional enhancement, not a requirement: without it the table shrinks to
what the standard library can answer (load average, disk usage, cpu count) and the rest
of the system behaves as though those channels do not exist, which is the honest report
rather than a zero.

Three filters keep the set usable, and each earns its place on a real macOS host, which
enumerates two dozen interfaces and eight APFS volumes of one container:

- **Interfaces**: up, non-loopback, and having carried traffic — then the busiest three.
- **Filesystems**: de-duplicated by observed capacity, because firmlinked volumes of one
  APFS container all report the *same* total and free bytes.
- **Everything**: a `DEFAULT_MAX_CHANNELS` valve, logged when it bites.

Nothing the host provider declares is writable. A discovered declaration carries no
envelope a person is accountable for, so `HostTransport.write` refuses every call and
reports `SIDE_EFFECT_NONE` — provable here, unlike in most transports, because the call
never reaches anything.

---

## 9. Out of scope

- **Audio playback.** A microphone exposes an input level scalar. A recording is a
  different capability with a different consequence, and this must not quietly become
  one: `FfmpegLevelReader` sends its output to `null` and only a number leaves it.
- **Emitting or egressing scanners.** Bluetooth (transmits) and mDNS/ONVIF (leaves the
  host) are not implemented here. They are a provider module plus a row, and they must be
  opt-in for the reason stated in §2.1.
- **Frames in stored history.** Media channels are never sampled. A trace of frames has
  no mean, and the payload ceiling on a finding is 256 KB.

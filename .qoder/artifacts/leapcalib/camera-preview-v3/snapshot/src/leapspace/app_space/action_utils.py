# Copyright (c) Alibaba, Inc. and its affiliates.
"""action_utils — in-sandbox X11/AT-SPI primitives driven via ``python -m``.

One module, three functions, dispatched by function name:

    python -m leapspace.app_space.action_utils unmap_overlay
    python -m leapspace.app_space.action_utils dump_elements <out_path>
    python -m leapspace.app_space.action_utils type_text <text>

Three gaps in the sandbox's input stack forced these helpers:

1. The driver paints a full-screen Cua.AgentCursorOverlay window stacked
   above the apps. Every X button event lands on it (X pointer events
   propagate to ancestors, never down to sibling windows below), so no
   pixel click ever reaches an app while it is mapped.
2. The driver's get_window_state returns only tree_markdown — no
   structured elements, no bounds — so click coordinates must come from
   an AT-SPI walk.
3. pynput's type() sends XSendEvent to the focused window: real text in
   the widget, but invisible to XRecord-based observers (the input tap
   records device events only). XTest fake_input produces true device
   events, visible to observers and delivered like a physical keyboard.

Every function imports its dependencies lazily: the module must stay
importable under both in-box interpreters — the repo venv python (pynput,
python-Xlib) drives unmap_overlay/type_text, while the OS python (apt
pyatspi) drives dump_elements.
"""

from __future__ import annotations

import sys


def unmap_overlay() -> None:
    """Unmap the driver's agent-cursor overlay windows; report the count."""
    import json

    from Xlib import display

    d = display.Display()
    unmapped = 0
    for child in d.screen().root.query_tree().children:
        try:
            name = child.get_wm_name() or ""
            if "Cua.AgentCursorOverlay" in name and child.get_attributes().map_state:
                child.unmap()
                unmapped += 1
        except Exception:
            # a window dying between query and read is not ours to handle
            pass
    d.flush()
    d.sync()
    print(json.dumps({"unmapped": unmapped}))


def dump_elements(out_path: str) -> None:
    """Walk the AT-SPI desktop tree; write named elements as JSONL.

    Each line carries name, role, screen-space bounds (XY_SCREEN), and the
    element's text value when it exposes one.
    """
    import json

    import pyatspi

    def value_of(obj):
        try:
            return obj.queryText().getText(0, -1)
        except Exception:
            return None

    def walk(obj, out):
        try:
            if obj.name:
                ext = obj.queryComponent().getExtents(pyatspi.XY_SCREEN)
                out.append(
                    {
                        "name": obj.name,
                        "role": obj.getRoleName(),
                        "x": ext.x,
                        "y": ext.y,
                        "w": ext.width,
                        "h": ext.height,
                        "value": value_of(obj),
                    }
                )
        except Exception:
            pass
        for index in range(obj.childCount):
            try:
                walk(obj.getChildAtIndex(index), out)
            except Exception:
                pass

    elements = []
    desktop = pyatspi.Registry.getDesktop(0)
    for index in range(desktop.childCount):
        walk(desktop.getChildAtIndex(index), elements)
    with open(out_path, "w") as fh:
        for element in elements:
            fh.write(json.dumps(element) + "\n")


def type_text(text: str) -> None:
    """Type text into the focused widget, one XTest keystroke per char.

    Latin-1 characters only; anything else exits non-zero naming the
    untypeable characters. Shift is held with 60ms spacing from its char so
    both keystrokes survive the input observer's 50ms per-action-type
    throttle.
    """
    import time

    from Xlib import X, XK, display
    from Xlib.ext import xtest
    from pynput._util.xorg import keyboard_mapping

    d = display.Display()
    mapping = keyboard_mapping(d)
    shift_keycode = mapping[XK.XK_Shift_L][0]
    skipped = []

    for ch in text:
        keysym = ord(ch) if 0x20 <= ord(ch) <= 0xFF else 0
        if not keysym or keysym not in mapping:
            skipped.append(ch)
            continue
        keycode, shift_state = mapping[keysym]
        if shift_state & 1:
            xtest.fake_input(d, X.KeyPress, shift_keycode)
            d.flush()
            time.sleep(0.06)
        xtest.fake_input(d, X.KeyPress, keycode)
        d.flush()
        time.sleep(0.04)
        xtest.fake_input(d, X.KeyRelease, keycode)
        d.flush()
        if shift_state & 1:
            xtest.fake_input(d, X.KeyRelease, shift_keycode)
            d.flush()
        time.sleep(0.06)
    d.sync()
    if skipped:
        sys.stderr.write("untypeable characters: " + repr("".join(skipped)) + "\n")
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``<function> [args...]`` to the named action_utils function."""
    argv = sys.argv[1:] if argv is None else argv
    functions = {
        "unmap_overlay": unmap_overlay,
        "dump_elements": dump_elements,
        "type_text": type_text,
    }
    if not argv:
        print(
            "usage: python -m leapspace.app_space.action_utils <function> [args...]; "
            f"functions: {', '.join(sorted(functions))}",
            file=sys.stderr,
        )
        return 2
    name, args = argv[0], argv[1:]
    fn = functions.get(name)
    if fn is None:
        print(
            f"unknown function {name!r}; must be one of {sorted(functions)}",
            file=sys.stderr,
        )
        return 2
    fn(*args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

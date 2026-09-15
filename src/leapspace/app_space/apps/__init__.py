# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapSpace scenario apps: the host-side app registry.

APP_MODULES is the only place an app_id resolves to its module: the
harness launches ``python -m <module>`` from it and lint whitelists
app_ids with it. Keep this package PyQt6-free — the harness and lint
import it on the host before any sandbox exists; the app classes live
in _base and the per-app modules.
"""

APP_MODULES: dict[str, str] = {
    "chat": "leapspace.app_space.apps.chat",
}

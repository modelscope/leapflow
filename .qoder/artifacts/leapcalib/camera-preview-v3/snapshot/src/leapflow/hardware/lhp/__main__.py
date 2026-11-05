# Copyright (c) Alibaba, Inc. and its affiliates.
"""Module entry: ``python -m leapflow.hardware.lhp --node-id <id>`` serves this
machine's hardware registry as an LHP node (the edge runtime; see ``runtime.py``).
"""

from __future__ import annotations

import sys

from leapflow.hardware.lhp.runtime import main

if __name__ == "__main__":
    sys.exit(main())

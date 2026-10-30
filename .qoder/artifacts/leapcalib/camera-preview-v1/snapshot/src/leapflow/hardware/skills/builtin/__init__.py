# Copyright (c) Alibaba, Inc. and its affiliates.
"""Built-in node skills shipped with LeapFlow.

Each module exposes ``skills()`` for the ``leapflow.hardware.skills`` entry-point
group. Built-ins obey the same dependency rule as third-party skills: they import
only ``leapflow.hardware.skills.protocol``, ``geometry``, and the standard library.
"""

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Device-specific drivers that are discovered through entry points.

Modules here are not generic transport mechanisms (those live in
``hardware/transports/``).  They wrap one particular hardware family's SDK
and are optional: their entry points are picked up only when the SDK is
installed, so an environment that does not own a camera or sensor never pays
for its driver's import errors.

The architecture contract in ``test_hardware_transports_are_not_named_after_one_device``
guards the boundary: a device-named module must live here, not in ``transports/``.
"""

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool implementations — the callable behaviour behind the agent's tools.

This package holds what tools *do* (file operations, shell, terminal sessions,
web fetch/extract, SCM, config, gateway dispatch, code intelligence) plus the
Tool Capability Contract in ``name_resolver``.

It deliberately exposes no registry: declaring tools, discovering plugins, and
owning the live catalog belong to ``leapflow.plugins``. Import
``leapflow.plugins.get_registry()`` to reach the assembled tool catalog.
"""

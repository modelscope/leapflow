"""Dashboard subsystem: declarative Server-Driven UI (SDUI) for monitoring.

This package owns the domain-neutral view layer:
- ``viewspec``: the validated component catalog and ViewSpec contract
- ``templates``: YAML template rendering into a ViewSpec (with safe binding)
- ``intent``: the ``DashboardIntent`` shared by slash and natural-language entry

Transport (the local web server) is added as a separate, optional module so the
core view logic stays importable without web dependencies.
"""

from leapflow.dashboard.intent import DashboardIntent
from leapflow.dashboard.hub import ViewHub
from leapflow.dashboard.service import (
    DaemonDataProvider,
    DashboardDataProvider,
    DashboardViewBuilder,
    select_template,
)
from leapflow.dashboard.templates import TemplateLibrary, render_template
from leapflow.dashboard.viewspec import (
    COMPONENT_CATALOG,
    COMPONENT_TYPES,
    SCHEMA_VERSION,
    normalize_viewspec,
    validate_viewspec,
)

__all__ = [
    "DashboardIntent",
    "ViewHub",
    "DashboardDataProvider",
    "DaemonDataProvider",
    "DashboardViewBuilder",
    "select_template",
    "TemplateLibrary",
    "render_template",
    "COMPONENT_CATALOG",
    "COMPONENT_TYPES",
    "SCHEMA_VERSION",
    "normalize_viewspec",
    "validate_viewspec",
]

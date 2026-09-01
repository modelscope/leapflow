"""app_space — the LeapSpace core: actor, task config, lint, and the apps.

Kept import-light on purpose: ``leapspace.app_space`` itself must import
without the ``leapspace`` dependency group (PyQt6 / cua-sandbox / pydantic);
only the submodules that need those dependencies pull them in.
"""

"""Scenario-based tests for platform synthesis (denoise + synthesis + intent inference).

Replaces granular rule-level tests in test_synthesis.py, test_denoise.py,
test_intent_inferrer.py, and test_patterns.py with end-to-end scenarios.
"""

from __future__ import annotations

from conftest import make_action
from leapflow.analysis.denoise import DenoisePass
from leapflow.analysis.synthesis import PlatformSynthesisPass
from leapflow.domain.trajectory import SemanticAction


def _shortcut(key_code: int, modifiers: list[str], raw_range=(0, 1)) -> SemanticAction:
    return SemanticAction(
        action_name="ui.shortcut",
        description="shortcut",
        parameters={"key_code": key_code, "modifiers": modifiers},
        raw_action_range=raw_range,
    )


def _rename(path: str, *, raw_range=(0, 1)) -> SemanticAction:
    return SemanticAction(
        action_name="file.rename",
        description=f"Rename {path}",
        parameters={"path": path, "target": path},
        raw_action_range=raw_range,
    )


# ═══════════════════════════════════════════════════════════════════
# Synthesis scenarios
# ═══════════════════════════════════════════════════════════════════


def test_synthesis_mkdir_and_batch_move() -> None:
    """mkdir + 6 rename pairs → file.create + batch_move(count=6)."""
    base = "/Users/jason/temp/leap_temp_work"
    actions = [
        make_action("file.create", params={"path": f"{base}/test_files"}, raw_range=(0, 1)),
        make_action("file.modify", params={"path": f"{base}/.fseventsd/00000123"}, raw_range=(1, 2)),
        _rename(f"{base}/a.pdf", raw_range=(2, 3)),
        _rename(f"{base}/test_files/a.pdf", raw_range=(3, 4)),
        _rename(f"{base}/b.pdf", raw_range=(4, 5)),
        _rename(f"{base}/test_files/b.pdf", raw_range=(5, 6)),
        _rename(f"{base}/c.pdf", raw_range=(6, 7)),
        _rename(f"{base}/test_files/c.pdf", raw_range=(7, 8)),
        _rename(f"{base}/d.pdf", raw_range=(8, 9)),
        _rename(f"{base}/test_files/d.pdf", raw_range=(9, 10)),
        _rename(f"{base}/e.pdf", raw_range=(10, 11)),
        _rename(f"{base}/test_files/e.pdf", raw_range=(11, 12)),
        _rename(f"{base}/f.pdf", raw_range=(12, 13)),
        _rename(f"{base}/test_files/f.pdf", raw_range=(13, 14)),
    ]

    result = PlatformSynthesisPass().apply(actions)

    assert len(result) == 2
    assert result[0].action_name == "file.create"
    assert result[0].parameters["path"] == f"{base}/test_files"
    assert result[1].action_name == "batch_move"
    assert result[1].parameters["count"] == 6
    assert result[1].parameters["target_dir"] == f"{base}/test_files"


def test_synthesis_system_noise_removed() -> None:
    """Platform-internal paths (.fseventsd, .DS_Store, .Spotlight) are filtered."""
    actions = [
        make_action("file.create", params={"path": "/Users/me/project"}),
        make_action("file.modify", params={"path": "/Users/me/.fseventsd/0000012345"}),
        make_action("file.modify", params={"path": "/Users/me/Documents/.DS_Store"}),
        make_action("file.create", params={"path": "/vol/.Spotlight-V100/index"}),
        make_action("file.rename", params={"path": "/Users/me/report.pdf"}),
    ]

    result = PlatformSynthesisPass().apply(actions)

    assert len(result) == 2
    assert result[0].parameters["path"] == "/Users/me/project"
    assert result[1].parameters["path"] == "/Users/me/report.pdf"


def test_synthesis_download_pattern() -> None:
    """file.create in Downloads with temp suffix → download_file."""
    actions = [
        make_action(
            "file.create",
            params={"path": "/Users/me/Downloads/report.pdf.crdownload"},
            raw_range=(0, 1),
        ),
        make_action(
            "file.modify",
            params={"path": "/Users/me/Downloads/report.pdf.crdownload"},
            raw_range=(1, 2),
        ),
        make_action(
            "file.rename",
            params={
                "path": "/Users/me/Downloads/report.pdf",
                "source": "/Users/me/Downloads/report.pdf.crdownload",
                "destination": "/Users/me/Downloads/report.pdf",
            },
            raw_range=(2, 3),
        ),
    ]

    result = PlatformSynthesisPass().apply(actions)

    assert len(result) == 1
    assert result[0].action_name == "download_file"
    assert result[0].parameters["path"] == "/Users/me/Downloads/report.pdf"


def test_synthesis_drag_drop_pattern() -> None:
    """ui.drag followed by file.move → drag_drop."""
    actions = [
        make_action(
            "ui.drag",
            params={"target": "invoice.pdf", "app_bundle_id": "com.apple.finder"},
            raw_range=(0, 1),
        ),
        make_action(
            "file.move",
            params={
                "source": "/Users/me/Desktop/invoice.pdf",
                "destination": "/Users/me/Archive/invoice.pdf",
            },
            raw_range=(1, 2),
        ),
    ]

    result = PlatformSynthesisPass().apply(actions)

    assert len(result) == 1
    assert result[0].action_name == "drag_drop"
    assert result[0].parameters["source"] == "/Users/me/Desktop/invoice.pdf"
    assert result[0].parameters["destination"] == "/Users/me/Archive/invoice.pdf"


def test_synthesis_window_management_removed() -> None:
    """Consecutive window move/resize noise collapses to a single window action."""
    actions = [
        make_action("file.create", params={"path": "/Users/me/doc.txt"}, raw_range=(0, 1)),
        make_action("ui.move", params={"target": "win1"}, raw_range=(1, 2)),
        make_action("ui.resize", params={"target": "win1"}, raw_range=(2, 3)),
        make_action("ui.move", params={"target": "win1"}, raw_range=(3, 4)),
        make_action("file.modify", params={"path": "/Users/me/doc.txt"}, raw_range=(4, 5)),
    ]

    result = PlatformSynthesisPass().apply(actions)

    assert len(result) == 3
    assert result[0].action_name == "file.create"
    assert result[1].action_name == "window.arrange"
    assert result[1].parameters["event_count"] == 3
    assert result[2].action_name == "file.modify"
    assert not any(a.action_name in ("ui.move", "ui.resize") for a in result)


def test_synthesis_preserves_meaningful_actions() -> None:
    """Real user actions pass through synthesis unchanged."""
    actions = [
        make_action("app.switch", params={"target": "com.apple.finder"}),
        make_action("clipboard.copy", params={"text": "hello world"}),
        make_action("ui.click", params={"target": "save_button"}),
        make_action("ui.type", params={"text": "draft content"}),
    ]

    result = PlatformSynthesisPass().apply(actions)

    assert len(result) == 4
    assert [a.action_name for a in result] == [
        "app.switch",
        "clipboard.copy",
        "ui.click",
        "ui.type",
    ]


def test_synthesis_gather_compose_pattern() -> None:
    """Multi-app copy-paste workflow → gather_and_compose."""
    actions = [
        make_action(
            "clipboard.copy",
            params={"app_bundle_id": "com.apple.Safari", "text": "Part A"},
            raw_range=(0, 1),
        ),
        make_action("app.switch", params={"app_bundle_id": "com.apple.Notes"}, raw_range=(1, 2)),
        make_action(
            "ui.shortcut",
            params={"target": "paste", "target_label": "Paste", "app_bundle_id": "com.apple.Notes"},
            raw_range=(2, 3),
        ),
        make_action("app.switch", params={"app_bundle_id": "com.google.Chrome"}, raw_range=(3, 4)),
        make_action(
            "clipboard.copy",
            params={"app_bundle_id": "com.google.Chrome", "text": "Part B"},
            raw_range=(4, 5),
        ),
        make_action("app.switch", params={"app_bundle_id": "com.apple.Notes"}, raw_range=(5, 6)),
        make_action(
            "ui.shortcut",
            params={"target": "paste", "target_label": "Paste", "app_bundle_id": "com.apple.Notes"},
            raw_range=(6, 7),
        ),
    ]

    result = PlatformSynthesisPass().apply(actions)

    assert len(result) == 1
    assert result[0].action_name == "gather_and_compose"
    assert result[0].parameters["fragment_count"] == 2
    assert result[0].parameters["target_app"] == "com.apple.Notes"


# ═══════════════════════════════════════════════════════════════════
# Denoise scenarios
# ═══════════════════════════════════════════════════════════════════


def test_denoise_undo_collapse() -> None:
    """Cmd+Z undo cancels the preceding action; both are removed."""
    undo = _shortcut(6, ["command"], raw_range=(2, 3))
    actions = [
        make_action("file.modify", params={"path": "/a.txt"}, raw_range=(0, 1)),
        make_action("ui.type", params={"text": "oops"}, raw_range=(1, 2)),
        undo,
    ]

    result = DenoisePass().apply(actions)

    assert len(result) == 1
    assert result[0].action_name == "file.modify"
    assert not any(a.action_name == "ui.shortcut" for a in result)


def test_denoise_idempotent_merge() -> None:
    """Repeated identical scroll actions are merged into one."""
    actions = [
        make_action("ui.scroll", params={"target": "main_window"}, raw_range=(0, 1)),
        make_action("ui.scroll", params={"target": "main_window"}, raw_range=(1, 2)),
        make_action("ui.scroll", params={"target": "main_window"}, raw_range=(2, 3)),
    ]

    result = DenoisePass().apply(actions)

    assert len(result) == 1
    assert result[0].action_name == "ui.scroll"
    assert result[0].parameters["_merged_count"] == 3


def test_denoise_full_pipeline() -> None:
    """Mixed undo + distraction noise → clean output."""
    undo = _shortcut(6, ["command"], raw_range=(3, 4))
    actions = [
        make_action("file.modify", params={"path": "/work.txt"}, raw_range=(0, 1)),
        make_action("ui.type", params={"text": "mistake"}, raw_range=(1, 2)),
        undo,
        make_action("ui.type", params={"text": "corrected"}, raw_range=(4, 5)),
        make_action(
            "app.switch",
            params={"target": "Safari", "_prev_app": "VSCode"},
            raw_range=(5, 6),
        ),
        make_action("ui.scroll", raw_range=(6, 7)),
        make_action("app.switch", params={"target": "VSCode"}, raw_range=(7, 8)),
        make_action("file.modify", params={"path": "/work.txt"}, raw_range=(8, 9)),
    ]

    result = DenoisePass().apply(actions)

    assert len(result) == 3
    assert result[0].action_name == "file.modify"
    assert result[1].action_name == "ui.type"
    assert result[1].parameters["text"] == "corrected"
    assert result[2].action_name == "file.modify"
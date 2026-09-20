"""Render a read-only demo console and a reviewable video edit decision list.

The renderer produces a local HTML evidence view and an edit decision list rather
than fabricating a movie from unavailable GUI footage. The backup-reel helper
concatenates supplied recordings with ffmpeg but never overwrites an output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from aaai_demo.evidence import EvidenceBundleError, sha256_file, viewspec_data


_STORYBOARD = (
    (0, 35, "Overview and evidence contract", "State L1/headless/archive boundaries before any success claim.", "mixed"),
    (35, 85, "Real headless structural seam", "Show the real PyQt rename, benign control, and opt-in admission.", "headless_structural"),
    (85, 130, "World model and OODA boundary", "Show architecture or archived record; do not report uncalibrated live accuracy.", "controlled"),
    (130, 235, "CE-X counterfactual matrix", "Show no-op, rejection, reuse, and acquisition hypothesis symmetrically.", "L1"),
    (235, 275, "Conditional signal archive", "Play only a complete archive with state/event evidence and expect() PASS.", "archive_conditional"),
    (275, 300, "Limitations and reproduction", "Link every claim to a hash and state unexercised boundaries.", "mixed"),
)


def _write_once(path: Path, content: str) -> None:
    """Write text atomically and reject accidental evidence replacement."""
    if path.exists():
        raise EvidenceBundleError(f"write-once render artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise EvidenceBundleError(f"cannot write render artifact: {path}") from exc


def _safe_json(value: Any) -> str:
    """Serialize data for an inline script without allowing a closing-script escape."""
    return json.dumps(value, indent=2, sort_keys=True).replace("</", "<\\/")


def _html(bundle: Mapping[str, Any]) -> str:
    """Return a self-contained four-column read-only evidence console."""
    payload = _safe_json(bundle)
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>LeapFlow AAAI Causal Trace</title>
<style>
:root {{ color-scheme:dark; --bg:#10141b; --panel:#18212d; --line:#344253; --text:#e8eef8; --muted:#a4b0c0; --ok:#66d9a6; --accent:#7ab7ff; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 Inter,system-ui,sans-serif; }}
header {{ padding:18px 24px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:24px; align-items:center; }} h1 {{ margin:0; font-size:20px; }} h2 {{ margin:0 0 10px; font-size:14px; color:var(--accent); letter-spacing:.04em; text-transform:uppercase; }}
small,.muted {{ color:var(--muted); }} .badge {{ border:1px solid var(--line); border-radius:999px; padding:4px 9px; color:var(--accent); white-space:nowrap; }}
.timeline {{ display:flex; gap:6px; padding:12px 24px; overflow:auto; border-bottom:1px solid var(--line); }} .tick {{ min-width:132px; color:var(--muted); border-left:2px solid var(--accent); padding-left:8px; }}
.grid {{ display:grid; grid-template-columns:2.2fr 1fr 1fr .8fr; gap:10px; padding:12px; min-height:54vh; }} .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; overflow:auto; }}
.card {{ border-top:1px solid var(--line); padding:10px 0; }} .card:first-of-type {{ border-top:0; }} .key {{ color:var(--muted); display:block; font-size:12px; }}
button {{ background:#243247; border:1px solid var(--line); color:var(--text); padding:7px 9px; border-radius:6px; cursor:pointer; margin:0 5px 6px 0; }} button:hover,button.active {{ border-color:var(--accent); color:var(--accent); }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:7px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-weight:500; }} .ok {{ color:var(--ok); }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
footer {{ padding:12px 24px 24px; color:var(--muted); }} @media(max-width:1000px) {{ .grid {{ grid-template-columns:1fr 1fr; }} }} @media(max-width:640px) {{ .grid {{ grid-template-columns:1fr; }} header {{ align-items:flex-start; flex-direction:column; }} }}
</style></head><body>
<header><div><h1>LeapFlow AAAI Demo · Causal Trace</h1><small id=\"run\"></small></div><span class=\"badge\" id=\"level\"></span></header>
<div class=\"timeline\"><div class=\"tick\">t0 · State snapshot</div><div class=\"tick\">t1 · Action</div><div class=\"tick\">t2 · Actual effect</div><div class=\"tick\">t3 · Independent oracle</div><div class=\"tick\">t4 · Teacher hindsight</div><div class=\"tick\">t5 · Governance</div><div class=\"tick\">t6 · Next turn</div></div>
<main class=\"grid\"><section class=\"panel\"><h2>Real Leapspace evidence</h2><div id=\"trajectory\"></div></section><section class=\"panel\"><h2>Observe / Orient</h2><div id=\"observe\"></div></section><section class=\"panel\"><h2>Decide / PCD</h2><div id=\"decide\"></div></section><section class=\"panel\"><h2>Governance / Effect</h2><div id=\"governance\"></div></section></main>
<section class=\"panel\" style=\"margin:0 12px\"><h2>CE-X four-arm matrix</h2><div id=\"arms\"></div></section><section class=\"panel\" style=\"margin:12px\"><h2>Evidence drawer</h2><div id=\"drawer\"></div></section>
<footer>Read-only derived view. It displays declared evidence boundaries rather than inferring unrecorded approval, installation, independent oracle, or daemon-runtime outcomes.</footer>
<script>
const BUNDLE = {payload};
const el = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>\"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const card = (key, value, cls='') => `<div class=\"card ${{cls}}\"><span class=\"key\">${{esc(key)}}</span>${{esc(value)}}</div>`;
const arms = Array.isArray(BUNDLE.arms) ? BUNDLE.arms : [];
let selected = arms[0] || null;
function render() {{
  el('run').textContent = `${{BUNDLE.run?.run_id || 'unknown run'}} · ${{BUNDLE.run?.protocol_id || 'unknown protocol'}}`;
  el('level').textContent = `Evidence level: ${{BUNDLE.evidence_level || 'unknown'}}`;
  const t = BUNDLE.trajectory || {{}};
  const archive = BUNDLE.signal_archive || {{}};
  const seam = BUNDLE.headless_seam || {{}};
  el('trajectory').innerHTML = card('Signal archive', archive.status || 'not_supplied') + card('Archive expect()', archive.expect?.verdict || 'not_recorded') + card('Archive artifacts', (archive.artifacts || []).length) + card('Legacy trajectory index', t.status || 'not_supplied') + card('Headless seam', seam.status || 'not_supplied');
  const trace = (BUNDLE.causal_trace || []).filter(x => x.arm === selected?.arm && x.stage === 'observe');
  el('observe').innerHTML = card('Drift relevance', selected?.drift_relevance) + card('Resolution', selected?.resolution) + card('Headless requirement', seam.requirement?.capability || 'not_supplied') + card('Benign control evidence', seam.benign_control?.evidence_count ?? 'not_supplied') + trace.map(x => card(x.title, x.summary)).join('');
  el('decide').innerHTML = card('Expected action', selected?.expected_action) + card('Actual action', selected?.actual_action) + card('Match to declared oracle', selected?.matches_oracle ? 'yes' : 'no', selected?.matches_oracle ? 'ok' : '') + card('Proposal / admitted', `${{selected?.proposal_count ?? 0}} / ${{selected?.admitted_count ?? 0}}`);
  el('governance').innerHTML = card('Approval', selected?.approval) + card('Registry mutation', selected?.registry_mutation) + card('Lifecycle', selected?.lifecycle) + card('Effect status', selected?.effect?.status) + card('Oracle boundary', selected?.effect?.note) + card('Not supported', (BUNDLE.claims?.not_supported || []).join(' · '));
  el('arms').innerHTML = `<p>${{arms.map(a => `<button class=\"${{a.arm === selected?.arm ? 'active' : ''}}\" data-arm=\"${{esc(a.arm)}}\">${{esc(a.arm)}}</button>`).join('')}}</p><table><thead><tr><th>Arm</th><th>Relevance</th><th>Resolution</th><th>Outcome</th><th>Registry</th><th>Oracle</th></tr></thead><tbody>${{arms.map(a => `<tr><td>${{esc(a.arm)}}</td><td>${{esc(a.drift_relevance)}}</td><td>${{esc(a.resolution)}}</td><td>${{esc(a.outcome)}}</td><td>${{esc(a.registry_mutation)}}</td><td>${{esc(a.effect?.independent_oracle)}}</td></tr>`).join('')}}</tbody></table>`;
  document.querySelectorAll('[data-arm]').forEach(button => button.onclick = () => {{ selected = arms.find(a => a.arm === button.dataset.arm) || selected; render(); }});
  el('drawer').innerHTML = `<pre>${{esc(JSON.stringify({{selected_arm:selected, claims:BUNDLE.claims, manifest:BUNDLE.run}}, null, 2))}}</pre>`;
}}
render();
</script></body></html>"""


def _storyboard(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Build a five-minute edit decision list with explicit evidence boundaries."""
    run = dict(bundle.get("run") or {})
    segments = [
        {
            "start_s": start,
            "end_s": end,
            "title": title,
            "narration_constraint": constraint,
            "evidence_level": evidence_level,
        }
        for start, end, title, constraint, evidence_level in _STORYBOARD
    ]
    return {
        "schema_version": 1,
        "kind": "aaai_demo_edit_decision_list",
        "run_id": run.get("run_id", ""),
        "duration_s": 300,
        "segments": segments,
        "required_corner_bug": "Evidence level and run ID must remain visible.",
        "prohibited_claims": list(dict(bundle.get("claims") or {}).get("not_supported") or []),
        "source_video": dict(bundle.get("trajectory") or {}).get("root", ""),
    }


def render_demo_package(bundle: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Create a write-once interactive evidence page and video composition inputs."""
    target = output_dir.expanduser().resolve()
    if target.exists():
        raise EvidenceBundleError(f"write-once render directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    try:
        index = target / "index.html"
        storyboard = target / "storyboard.json"
        viewspec = target / "causal-trace.viewspec-data.json"
        _write_once(index, _html(bundle))
        _write_once(storyboard, json.dumps(_storyboard(bundle), indent=2, sort_keys=True) + "\n")
        _write_once(viewspec, json.dumps(viewspec_data(bundle), indent=2, sort_keys=True) + "\n")
        checksums = target / "checksums.sha256"
        rows = [f"{sha256_file(path)}  {path.name}" for path in sorted(target.iterdir()) if path.is_file()]
        _write_once(checksums, "\n".join(rows) + "\n")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"index": index, "storyboard": storyboard, "viewspec": viewspec, "checksums": checksums}


def compose_backup_reel(clips: Sequence[Path], output_path: Path) -> Path:
    """Concatenate reviewed video clips with ffmpeg while preserving their pixels."""
    if not clips:
        raise EvidenceBundleError("at least one reviewed video clip is required")
    if output_path.exists():
        raise EvidenceBundleError(f"write-once video output already exists: {output_path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise EvidenceBundleError("ffmpeg is unavailable; render the HTML and EDL instead")
    resolved = [clip.expanduser().resolve() for clip in clips]
    if any(not clip.is_file() for clip in resolved):
        raise EvidenceBundleError("a requested backup clip does not exist")
    concat = output_path.with_suffix(".concat.txt")
    if concat.exists():
        raise EvidenceBundleError(f"write-once concat list already exists: {concat}")
    concat.parent.mkdir(parents=True, exist_ok=True)
    lines = ["file '" + str(clip).replace("'", "'\\''") + "'" for clip in resolved]
    _write_once(concat, "\n".join(lines) + "\n")
    command = [ffmpeg, "-n", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(output_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceBundleError(f"ffmpeg could not compose backup reel: {exc}") from exc
    return output_path


__all__ = ["compose_backup_reel", "render_demo_package"]

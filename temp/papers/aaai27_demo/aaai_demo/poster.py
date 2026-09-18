"""Create a provisional, editable poster source for an accepted demo station.

AAAI-27 supplies a poster board only after acceptance and defers exhibit format
information to the Chairs. This module therefore produces a responsive 3:4 design
source, not an asserted print size or a submission replacement for the required
video/slides material.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from aaai_demo.evidence import EvidenceBundleError, sha256_file


_PROVISIONAL_FORMAT = {
    "status": "provisional",
    "design_ratio": "3:4 portrait",
    "official_size": "pending_chairs_exhibit_format_information",
    "print_export": "deferred_until_official_format_is_confirmed",
}


def _write_once(path: Path, content: str) -> None:
    """Write a source artifact atomically without replacing an existing one."""
    if path.exists():
        raise EvidenceBundleError(f"write-once poster artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise EvidenceBundleError(f"cannot write poster artifact: {path}") from exc


def _safe_json(value: Any) -> str:
    """Embed data without allowing a script closing tag to escape the document."""
    return json.dumps(value, indent=2, sort_keys=True).replace("</", "<\\/")


def _poster_html(bundle: Mapping[str, Any], viewer_url: str) -> str:
    """Render a vector-friendly poster source inspired by the supplied hierarchy."""
    payload = _safe_json(bundle)
    target = str(viewer_url or "SET_AFTER_ANONYMITY_REVIEW")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeapFlow Demonstration Poster Source</title>
<style>
:root {{ --ink:#173b69; --blue:#2f5f95; --pale:#eaf1f8; --line:#9eb3c8; --text:#18222d; --muted:#566575; --l1:#396b99; --headless:#956b20; --archive:#3e7b68; --absent:#9a3d3d; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#dfe6ed; color:var(--text); font:15px/1.35 Arial,Helvetica,sans-serif; }}
.poster {{ width:min(100vw,900px); min-height:1200px; margin:auto; padding:30px; background:#fff; display:grid; grid-template-columns:1.1fr .9fr; gap:18px; }}
header {{ grid-column:1/-1; border-bottom:4px solid var(--ink); padding-bottom:15px; position:relative; }} h1 {{ margin:0; color:var(--ink); font-size:38px; line-height:1.05; letter-spacing:.01em; }} .thesis {{ margin:10px 180px 0 0; color:var(--muted); font-size:17px; }}
.format {{ position:absolute; right:0; top:0; max-width:160px; border:1px solid var(--line); padding:8px; color:var(--muted); font-size:11px; }}
.panel {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#fff; }} .panel h2 {{ margin:0; padding:8px 12px; background:linear-gradient(90deg,var(--blue),#8ba9c7); color:#fff; font-size:18px; }} .body {{ padding:12px; }}
.wide {{ grid-column:1/-1; }} .flow {{ display:flex; flex-wrap:wrap; align-items:center; gap:7px; }} .node {{ padding:6px 8px; border:1px solid var(--line); background:var(--pale); font-size:12px; }} .arrow {{ color:var(--blue); font-weight:bold; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }} .tag {{ border:1px solid var(--line); padding:3px 6px; font-size:11px; }} .l1 {{ border-color:var(--l1); color:var(--l1); }} .headless {{ border-color:var(--headless); color:var(--headless); }} .archive {{ border-color:var(--archive); color:var(--archive); }} .absent {{ border-color:var(--absent); color:var(--absent); }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ text-align:left; border-bottom:1px solid #d9e1e9; padding:5px; vertical-align:top; }} th {{ color:var(--ink); }} .metric {{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }} .metric div {{ background:var(--pale); padding:8px; }} .metric b {{ display:block; color:var(--ink); font-size:20px; }}
.qr {{ min-height:118px; border:2px dashed var(--line); display:flex; align-items:center; justify-content:center; text-align:center; color:var(--muted); font-size:12px; padding:12px; }} code {{ overflow-wrap:anywhere; font-size:10px; }} .limit {{ border-left:5px solid var(--absent); padding:8px 10px; background:#fff5f5; }}
footer {{ grid-column:1/-1; border-top:2px dashed var(--line); padding-top:10px; color:var(--muted); font-size:11px; }} @media print {{ body {{ background:#fff; }} .poster {{ width:100%; min-height:0; }} }}
</style></head><body><main class="poster">
<header><h1>LeapFlow: From Environment Signal to Governed Capability Decision</h1><p class="thesis">A static guide to the live Harness demonstration. Every claim is bound to an auditable evidence lane; missing steps remain visible rather than being inferred.</p><div class="format">PROVISIONAL SOURCE<br>3:4 design sketch only<br>Final size pending Chairs</div></header>
<section class="panel"><h2>1. WHY A HARNESS NEEDS EVIDENCE</h2><div class="body"><p>Environmental drift is a hypothesis, not an instruction to rewrite capability. The demonstration makes baseline/no-op, irrelevant/reject, catalog reuse, and acquisition hypotheses equally visible.</p><div class="legend"><span class="tag l1">L1 controlled CE-X</span><span class="tag headless">real headless seam</span><span class="tag archive">conditional archive</span><span class="tag absent">not exercised</span></div></div></section>
<section class="panel"><h2>4. QUANTIFIED COUNTERFACTUALS</h2><div class="body"><table id="arms"></table><div class="metric" id="metrics"></div></div></section>
<section class="panel"><h2>2. SIGNAL CONTRACT</h2><div class="body"><div class="flow"><span class="node">real offscreen widget</span><span class="arrow">→</span><span class="node">role-aware elements</span><span class="arrow">→</span><span class="node">InterfaceDelta</span><span class="arrow">→</span><span class="node">opt-in environment_probe requirement</span></div><p>Benign additions remain no-op. The headless lane has no framebuffer and is not GUI-agent e2e.</p></div></section>
<section class="panel"><h2>5. LIVE INTERACTION</h2><div class="body"><ol><li>Open the evidence viewer.</li><li>Select an episode or CE-X arm.</li><li>Observe Environment / Decision / Governance lanes in Evolution Live.</li></ol><div class="qr">QR target is set only after anonymity review:<br><code>{target}</code></div></div></section>
<section class="panel wide"><h2>3. DECISION AND GOVERNANCE</h2><div class="body"><div class="flow"><span class="node">observe</span><span class="arrow">→</span><span class="node">requirement</span><span class="arrow">→</span><span class="node">resolution</span><span class="arrow">→</span><span class="node">absorb / rebind / acquire / escalate</span><span class="arrow">→</span><span class="node">policy / lifecycle</span></div><p>Acquire is a hypothesis. Approval, install, daemon-runtime trust, and independent oracle outcomes are shown only when a same-run artifact exists.</p></div></section>
<section class="panel wide"><h2>6. CLAIMS AND LIMITS</h2><div class="body"><div class="limit" id="limits"></div><p>Reproduce: <code>headless-seam → run_c2 → bundle → render</code>. The monitor hosts dynamic replay; this poster is only the static entry point.</p></div></section>
<footer>Poster source is not a submission replacement. The AAAI-27 initial submission requires a short paper plus a ≤5-minute video or slides. Generate the final printed artifact only after official exhibit format information is supplied.</footer>
</main><script>
const BUNDLE = {payload};
const esc = value => String(value ?? '').replace(/[&<>\"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const arms = Array.isArray(BUNDLE.arms) ? BUNDLE.arms : [];
document.getElementById('arms').innerHTML = '<thead><tr><th>Arm</th><th>Expected / actual</th><th>Evidence boundary</th></tr></thead><tbody>' + arms.map(a => '<tr><td>' + esc(a.arm) + '</td><td>' + esc(a.expected_action) + ' / ' + esc(a.actual_action) + '</td><td>' + esc(a.effect?.status || 'not_recorded') + '</td></tr>').join('') + '</tbody>';
const summary = BUNDLE.summary || {{}}; document.getElementById('metrics').innerHTML = [['Arms',summary.arms_total || arms.length],['Correct',summary.arms_correct || 0],['Mutations',summary.mutations || 0]].map(p => '<div><b>' + esc(p[1]) + '</b>' + esc(p[0]) + '</div>').join('');
document.getElementById('limits').innerHTML = '<b>Not supported in this bundle:</b><br>' + esc((BUNDLE.claims?.not_supported || []).join(' · '));
</script></body></html>"""


def render_poster_source(
    bundle: Mapping[str, Any], output_dir: Path, *, viewer_url: str = ""
) -> dict[str, Path]:
    """Write a provisional poster source and its evidence data without overwriting."""
    if bundle.get("kind") != "aaai_demo_evidence_bundle":
        raise EvidenceBundleError("poster source requires an AAAI demo evidence bundle")
    target = output_dir.expanduser().resolve()
    if target.exists():
        raise EvidenceBundleError(f"write-once poster directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    try:
        poster = target / "poster.html"
        data = target / "poster-data.json"
        format_note = target / "format-status.json"
        _write_once(poster, _poster_html(bundle, viewer_url))
        _write_once(data, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        _write_once(format_note, json.dumps(_PROVISIONAL_FORMAT, indent=2, sort_keys=True) + "\n")
        checksums = target / "checksums.sha256"
        lines = [
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(target.iterdir())
            if path.is_file()
        ]
        _write_once(checksums, "\n".join(lines) + "\n")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"poster": poster, "data": data, "format": format_note, "checksums": checksums}


__all__ = ["render_poster_source"]

"""HTML report generator for the drug-discovery agent evaluation suite.

Usage (called automatically by run_eval.py --html, or standalone):

    from eval.report_html import generate_html
    html = generate_html(records, agg, model, tag)
    open("report.html", "w").write(html)
"""
from __future__ import annotations
from datetime import datetime, timezone
import html as _html
import json


_VERDICT_COLOR = {
    "PASS":    ("#d1fae5", "#065f46"),   # green
    "FAIL":    ("#fee2e2", "#991b1b"),   # red
    "ERROR":   ("#fef3c7", "#92400e"),   # amber
    "REVIEW":  ("#dbeafe", "#1e3a8a"),   # blue
    "REVIEW*": ("#ede9fe", "#4c1d95"),   # purple
}


def _verdict_style(verdict: str) -> str:
    bg, fg = _VERDICT_COLOR.get(verdict.split()[0], ("#f3f4f6", "#111827"))
    return f"background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.8rem;"


def _e(s) -> str:
    return _html.escape(str(s or ""))


def _answer_cell(answer: str) -> str:
    """Collapsible answer cell — show first 120 chars, expand on click."""
    if not answer:
        return "<em style='color:#6b7280'>—</em>"
    short = _e(answer[:120]) + ("…" if len(answer) > 120 else "")
    full = _e(answer)
    uid = id(answer)
    return (
        f"<span id='s{uid}'>{short} "
        f"<a href='#' onclick=\"document.getElementById('s{uid}').style.display='none';"
        f"document.getElementById('f{uid}').style.display='block';return false;\" "
        f"style='font-size:0.75rem;color:#6366f1'>[show]</a></span>"
        f"<span id='f{uid}' style='display:none;white-space:pre-wrap'>{full} "
        f"<a href='#' onclick=\"document.getElementById('f{uid}').style.display='none';"
        f"document.getElementById('s{uid}').style.display='inline';return false;\" "
        f"style='font-size:0.75rem;color:#6366f1'>[hide]</a></span>"
    )


def _failures_cell(failures: list) -> str:
    if not failures:
        return "<span style='color:#10b981'>✓ all checks passed</span>"
    items = "".join(f"<li>{_e(f)}</li>" for f in failures)
    return f"<ul style='margin:0;padding-left:1.2em;color:#991b1b;font-size:0.8rem'>{items}</ul>"


def _tools_cell(tools: list) -> str:
    if not tools:
        return "<em style='color:#6b7280'>none</em>"
    badges = "".join(
        f"<span style='background:#e0e7ff;color:#3730a3;padding:1px 6px;border-radius:3px;"
        f"font-size:0.75rem;margin:1px;display:inline-block'>{_e(t)}</span>"
        for t in dict.fromkeys(tools)  # deduplicate, preserve order
    )
    return badges


def _summary_card(label: str, value: str, sub: str = "", color: str = "#6366f1") -> str:
    return (
        f"<div style='background:#fff;border:1px solid #e5e7eb;border-radius:8px;"
        f"padding:16px 20px;min-width:160px;flex:1'>"
        f"<div style='font-size:0.75rem;color:#6b7280;text-transform:uppercase;"
        f"letter-spacing:.05em'>{label}</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{color}'>{value}</div>"
        f"<div style='font-size:0.75rem;color:#9ca3af'>{sub}</div>"
        f"</div>"
    )


def _category_breakdown(records: list[dict]) -> str:
    from collections import defaultdict
    cats: dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "review": 0})
    for r in records:
        cat = r.get("category", "?")
        v = r.get("verdict", "")
        if v == "PASS":
            cats[cat]["pass"] += 1
        elif v in ("REVIEW", "REVIEW*"):
            cats[cat]["review"] += 1
        elif v == "ERROR":
            cats[cat]["error"] += 1
        else:
            cats[cat]["fail"] += 1

    rows = ""
    for cat, counts in sorted(cats.items()):
        total = sum(counts.values())
        pct = int(100 * counts["pass"] / total) if total else 0
        bar_color = "#10b981" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
        rows += (
            f"<tr>"
            f"<td style='padding:6px 12px;text-transform:capitalize'>{_e(cat)}</td>"
            f"<td style='padding:6px 12px;text-align:center'>{total}</td>"
            f"<td style='padding:6px 12px;text-align:center;color:#10b981'>{counts['pass']}</td>"
            f"<td style='padding:6px 12px;text-align:center;color:#ef4444'>{counts['fail']}</td>"
            f"<td style='padding:6px 12px;text-align:center;color:#92400e'>{counts['error']}</td>"
            f"<td style='padding:6px 12px;text-align:center;color:#1e3a8a'>{counts['review']}</td>"
            f"<td style='padding:6px 12px'>"
            f"  <div style='background:#f3f4f6;border-radius:4px;height:12px;width:100%;min-width:80px'>"
            f"    <div style='background:{bar_color};height:12px;border-radius:4px;width:{pct}%'></div>"
            f"  </div>"
            f"  <span style='font-size:0.75rem;color:#6b7280'>{pct}%</span>"
            f"</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:0.875rem'>"
        "<thead><tr style='background:#f9fafb'>"
        "<th style='padding:8px 12px;text-align:left;border-bottom:1px solid #e5e7eb'>Category</th>"
        "<th style='padding:8px 12px;text-align:center;border-bottom:1px solid #e5e7eb'>Total</th>"
        "<th style='padding:8px 12px;text-align:center;border-bottom:1px solid #e5e7eb'>Pass</th>"
        "<th style='padding:8px 12px;text-align:center;border-bottom:1px solid #e5e7eb'>Fail</th>"
        "<th style='padding:8px 12px;text-align:center;border-bottom:1px solid #e5e7eb'>Error</th>"
        "<th style='padding:8px 12px;text-align:center;border-bottom:1px solid #e5e7eb'>Review</th>"
        "<th style='padding:8px 12px;border-bottom:1px solid #e5e7eb'>Pass rate</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def generate_html(records: list[dict], agg: dict, model: str, tag: str | None) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_total = agg.get("n_total", len(records))
    auto_pass = agg.get("auto_pass", 0)
    auto_n = agg.get("n_auto", 1)
    auto_rate = agg.get("auto_pass_rate", f"{100*auto_pass//auto_n}%" if auto_n else "n/a")
    halluc_pass = agg.get("halluc_pass", 0)
    halluc_n = agg.get("halluc_total", 1)
    halluc_rate = agg.get("halluc_pass_rate", "n/a")
    errors = agg.get("n_errors", 0)
    guard = agg.get("guard_interventions", 0)
    lat_mean = agg.get("latency_mean", "n/a")
    lat_p95 = agg.get("latency_p95", "n/a")
    hops = agg.get("hops_mean", "n/a")

    # Failures that need attention
    failures = [r for r in records if r.get("verdict") not in ("PASS", "REVIEW")]

    # Detail rows
    detail_rows = ""
    for r in records:
        verdict = r.get("verdict", "?")
        vbg, vfg = _VERDICT_COLOR.get(verdict.split()[0], ("#f3f4f6", "#111827"))
        border = "border-left:4px solid #ef4444;" if verdict not in ("PASS", "REVIEW") else "border-left:4px solid #10b981;"
        detail_rows += (
            f"<tr style='vertical-align:top;border-bottom:1px solid #f3f4f6;{border}'>"
            f"<td style='padding:8px 10px;white-space:nowrap;font-size:0.78rem;color:#6b7280'>{_e(r['id'])}</td>"
            f"<td style='padding:8px 10px;white-space:nowrap'>"
            f"  <span style='background:#f3f4f6;padding:1px 6px;border-radius:3px;font-size:0.75rem'>{_e(r.get('category',''))}</span>"
            f"</td>"
            f"<td style='padding:8px 10px;font-size:0.875rem;max-width:280px'>{_e(r.get('question',''))}</td>"
            f"<td style='padding:8px 10px;text-align:center'>"
            f"  <span style='{_verdict_style(verdict)}'>{_e(verdict)}</span>"
            f"</td>"
            f"<td style='padding:8px 10px;text-align:right;font-size:0.8rem;white-space:nowrap'>{r.get('latency_s',''):.1f}s" if isinstance(r.get('latency_s'), float) else f"<td style='padding:8px 10px;text-align:right;font-size:0.8rem'>{r.get('latency_s','')}"
            f"</td>"
            f"<td style='padding:8px 10px'>{_tools_cell(r.get('tools_called', []))}</td>"
            f"<td style='padding:8px 10px;max-width:260px'>{_failures_cell(r.get('failures', []))}</td>"
            f"<td style='padding:8px 10px;max-width:320px;font-size:0.8rem'>{_answer_cell(r.get('answer',''))}</td>"
            f"</tr>"
        )

    # Problems section
    problems_html = ""
    if failures:
        prob_rows = ""
        for r in failures:
            verdict = r.get("verdict", "?")
            prob_rows += (
                f"<div style='background:#fff;border:1px solid #fca5a5;border-radius:6px;padding:14px;margin-bottom:10px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:8px'>"
                f"<div>"
                f"  <span style='font-weight:600;font-size:0.875rem'>{_e(r['id'])}</span>"
                f"  <span style='margin-left:8px;background:#f3f4f6;padding:1px 6px;border-radius:3px;font-size:0.75rem'>{_e(r.get('category',''))}</span>"
                f"</div>"
                f"<span style='{_verdict_style(verdict)}'>{_e(verdict)}</span>"
                f"</div>"
                f"<div style='margin-top:6px;font-size:0.875rem;color:#374151'>{_e(r.get('question',''))}</div>"
                f"<div style='margin-top:8px'>{_failures_cell(r.get('failures', []))}</div>"
                f"<div style='margin-top:6px;font-size:0.78rem;color:#6b7280'>Tools called: {_tools_cell(r.get('tools_called', []))}</div>"
                f"</div>"
            )
        problems_html = (
            f"<section style='margin-bottom:32px'>"
            f"<h2 style='font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#991b1b'>"
            f"⚠ Problems requiring attention ({len(failures)})</h2>"
            f"{prob_rows}"
            f"</section>"
        )
    else:
        problems_html = (
            "<section style='margin-bottom:32px'>"
            "<div style='background:#d1fae5;border:1px solid #6ee7b7;border-radius:6px;padding:14px;color:#065f46;font-weight:600'>"
            "✓ All automated checks passed — no failures detected."
            "</div></section>"
        )

    title = f"AI Drug Discovery — Evaluation Report" + (f" [{tag}]" if tag else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f9fafb; color: #111827; line-height: 1.5; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 12px; color: #374151; }}
  .meta {{ font-size: 0.8rem; color: #6b7280; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }}
  section {{ margin-bottom: 28px; }}
  .section-box {{ background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:20px; }}
  table.detail {{ width:100%;border-collapse:collapse;font-size:0.82rem; }}
  table.detail thead tr {{ background:#f9fafb; }}
  table.detail thead th {{ padding:8px 10px;text-align:left;border-bottom:2px solid #e5e7eb;white-space:nowrap; }}
  .filter-bar {{ display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px; }}
  .filter-btn {{ background:#f3f4f6;border:1px solid #d1d5db;border-radius:4px;padding:4px 12px;
                 cursor:pointer;font-size:0.8rem; }}
  .filter-btn.active {{ background:#4f46e5;color:#fff;border-color:#4f46e5; }}
</style>
</head>
<body>
<div class="container">
  <h1>{_e(title)}</h1>
  <p class="meta">Model: <strong>{_e(model)}</strong> &nbsp;|&nbsp; {when} &nbsp;|&nbsp; {n_total} questions</p>

  <!-- Summary cards -->
  <div class="cards">
    {_summary_card("Auto pass rate", str(auto_rate), f"{auto_pass}/{auto_n} questions",
                   "#10b981" if isinstance(auto_rate, str) and "%" in str(auto_rate) and float(str(auto_rate).replace("%",""))>=80 else "#f59e0b")}
    {_summary_card("Hallucination probes", str(halluc_rate), f"{halluc_pass}/{halluc_n} blocked correctly", "#6366f1")}
    {_summary_card("Errors", str(errors), "agent crashes / timeouts", "#ef4444" if errors else "#10b981")}
    {_summary_card("Guard interventions", str(guard), "ungrounded SMILES stripped", "#f59e0b" if guard else "#10b981")}
    {_summary_card("Avg latency", f"{lat_mean}s", f"p95 = {lat_p95}s", "#6366f1")}
    {_summary_card("Avg LLM hops", str(hops), "round-trips per question", "#6366f1")}
  </div>

  <!-- Category breakdown -->
  <section class="section-box" style="margin-bottom:28px">
    <h2>Pass rate by category</h2>
    {_category_breakdown(records)}
  </section>

  <!-- Problems -->
  {problems_html}

  <!-- Full detail table -->
  <section class="section-box">
    <h2>All questions</h2>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="filterTable('ALL', this)">All</button>
      <button class="filter-btn" onclick="filterTable('PASS', this)">Pass</button>
      <button class="filter-btn" onclick="filterTable('FAIL', this)">Fail</button>
      <button class="filter-btn" onclick="filterTable('ERROR', this)">Error</button>
      <button class="filter-btn" onclick="filterTable('REVIEW', this)">Review</button>
    </div>
    <div style="overflow-x:auto">
      <table class="detail" id="detail-table">
        <thead>
          <tr>
            <th>ID</th><th>Category</th><th>Question</th>
            <th>Verdict</th><th>Latency</th><th>Tools called</th>
            <th>Failures</th><th>Answer preview</th>
          </tr>
        </thead>
        <tbody>
          {detail_rows}
        </tbody>
      </table>
    </div>
  </section>

  <p style="margin-top:16px;font-size:0.75rem;color:#9ca3af">
    REVIEW = manual inspection needed (automated checks ran but cannot fully verify).
    REVIEW* = manual item that also failed an automated check.
    Guard interventions = ungrounded SMILES the SMILES guard would strip from the live API response;
    rubric scores are on raw model output so guard cannot hide hallucinations.
  </p>
</div>

<script>
function filterTable(verdict, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#detail-table tbody tr').forEach(row => {{
    if (verdict === 'ALL') {{ row.style.display = ''; return; }}
    const cell = row.querySelector('td:nth-child(4)');
    const v = cell ? cell.innerText.trim() : '';
    row.style.display = v.startsWith(verdict) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

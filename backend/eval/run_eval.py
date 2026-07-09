"""End-to-end evaluation harness for the drug-discovery agent.

Runs every question in dataset.json through the live LangGraph orchestrator,
capturing for each one:
  - the final answer text,
  - which tools were called (routing),
  - how many LLM round-trips the manager made (the real driver of latency),
  - wall-clock latency.

It then scores answers against the per-question rubric (eval/checks.py) and
writes two artifacts:
  - eval_report.txt   human-readable summary table + aggregates (thesis-ready),
  - eval_results.json raw per-question records (for plots / deeper analysis).

This is the BASELINE measurement of the current single-agent system. Re-running
it after each architecture change (multi-agent, model routing, generative agent)
produces the comparison tables the thesis is built on. The manager model is read
from MANAGER_MODEL, so the same suite drives the model-size ablation:

    # baseline
    python -m eval.run_eval
    # 7b vs 14b ablation
    MANAGER_MODEL=qwen2.5:7b  python -m eval.run_eval --tag 7b
    MANAGER_MODEL=qwen2.5:14b python -m eval.run_eval --tag 14b

Requires the same backing services as the app (Ollama with the chosen model
pulled, Neo4j running with the RAG index built). Run from the backend/ directory.
"""
import os
import sys
import json
import time
import argparse
import statistics
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.checks import evaluate
from eval.stats import mean_ci, wilson_ci
from core.config import MANAGER_MODEL

HERE = os.path.dirname(__file__)
DATASET_PATH = os.path.join(HERE, "dataset.json")  # default; overridden by --dataset
RECURSION_LIMIT = 15


def _run_question(question: str):
    """Stream one question through the orchestrator, returning the captured trace.

    Imported lazily so --help and arg parsing work without spinning up Ollama.
    """
    from agents.orchestrator import orchestrator
    from agents import trace, smiles_guard

    inputs = {"messages": [("user", question)]}
    tools_called: list[str] = []
    llm_hops = 0
    final = ""
    error = None

    # The top-level stream only shows which SPECIALIST (sub-agent) the supervisor
    # routed to. Each specialist records its INNER tool calls here so routing
    # checks that reference low-level tools (search_pubmed, predict_toxicity, …)
    # still see them — keeping the baseline comparison apples-to-apples.
    trace.reset()
    # Start a fresh SMILES-provenance window and seed it with anything the user
    # supplied, so the guard can strip ungrounded structures from the answer.
    smiles_guard.reset()
    smiles_guard.record_user_message(question)

    t0 = time.perf_counter()
    try:
        for chunk in orchestrator.stream(
            inputs,
            stream_mode="updates",
            config={"recursion_limit": RECURSION_LIMIT},
        ):
            for _node, payload in chunk.items():
                for msg in payload.get("messages", []):
                    if getattr(msg, "tool_calls", None):
                        llm_hops += 1
                        tools_called.extend(tc["name"] for tc in msg.tool_calls)
                    elif getattr(msg, "type", None) == "ai" and msg.content:
                        llm_hops += 1
                        final = msg.content
    except Exception as e:  # noqa: BLE001 - record, don't crash the whole suite
        error = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0

    # Merge the specialists' inner tool calls in alongside the supervisor's
    # agent-level routing, so both routing layers are visible to the rubric.
    tools_called.extend(trace.get())

    # IMPORTANT: score the RAW model output, never the guarded one. The rubric
    # (incl. the hallucination probes that forbid ungrounded SMILES) must measure
    # what the MODEL did; sanitising first would let the guard hide a real
    # hallucination and turn a fail into a pass. We separately record what the
    # live guard WOULD strip, so the guard's defense-in-depth value is reported
    # honestly instead of masking model behaviour.
    answer_delivered, guard_removed = smiles_guard.sanitize(final)

    return {
        "answer": final,                    # raw model output — what the rubric scores
        "answer_delivered": answer_delivered,  # what the live API would return to the user
        "guard_removed": guard_removed,        # ungrounded SMILES the guard stripped
        "tools_called": tools_called,
        "llm_hops": llm_hops,
        "latency_s": round(elapsed, 2),
        "error": error,
    }


def _run_pass(questions: list[dict]) -> list[dict]:
    """Run every question once and return the scored per-question records."""
    records = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} ...", end=" ", flush=True)
        trace = _run_question(q["question"])
        passed, failures = evaluate(q.get("checks", {}), trace["answer"], trace["tools_called"])
        manual = q.get("manual_review", False)

        if trace["error"]:
            verdict = "ERROR"
        elif manual:
            verdict = "REVIEW" if passed else "REVIEW*"
        else:
            verdict = "PASS" if passed else "FAIL"

        print(f"{verdict}  ({trace['latency_s']}s, {trace['llm_hops']} hops)")

        records.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "manual_review": manual,
            "passed": passed,
            "failures": failures,
            "verdict": verdict,
            **trace,
        })

    return records


def run_suite(tag: str | None, runs: int = 1) -> list[list[dict]]:
    """Run the whole suite `runs` times, returning one record list per pass.

    Multiple passes are what make the metrics defensible: an LLM-backed pipeline
    is not bit-reproducible even at temperature 0, so we report each headline
    number with a confidence interval across passes rather than a single value.
    """
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    questions = dataset["questions"]

    print(f"Running {len(questions)} questions on model '{MANAGER_MODEL}'"
          + (f" (tag: {tag})" if tag else "")
          + (f"  ×{runs} runs" if runs > 1 else "") + "\n")

    # Build the orchestrator graph (initialises _llm_supervisor etc.) then warm up.
    from agents.specialists import build_specialists
    from agents.orchestrator import build_orchestrator, warmup
    build_specialists()
    build_orchestrator()
    print("Warming up model (loading into Ollama)...", flush=True)
    warmup()

    passes = []
    for r in range(1, runs + 1):
        if runs > 1:
            print(f"\n----- run {r}/{runs} -----")
        passes.append(_run_pass(questions))
    return passes


def _aggregate(records):
    def pct(num, den):
        return f"{100.0 * num / den:.1f}%" if den else "n/a"

    auto = [r for r in records if not r["manual_review"] and not r["error"]]
    auto_pass = sum(r["passed"] for r in auto)
    errors = [r for r in records if r["error"]]
    latencies = [r["latency_s"] for r in records if not r["error"]]
    hops = [r["llm_hops"] for r in records if not r["error"]]

    halluc = [r for r in records if r["category"] == "hallucination" and not r["error"]]
    halluc_pass = sum(r["passed"] for r in halluc)

    lat_sorted = sorted(latencies)

    def pctl(p):
        if not lat_sorted:
            return float("nan")
        k = max(0, min(len(lat_sorted) - 1, int(round((p / 100) * (len(lat_sorted) - 1)))))
        return lat_sorted[k]

    return {
        "n_total": len(records),
        "n_auto": len(auto),
        "auto_pass": auto_pass,
        "auto_pass_rate": pct(auto_pass, len(auto)),
        "n_errors": len(errors),
        "halluc_total": len(halluc),
        "halluc_pass": halluc_pass,
        "halluc_pass_rate": pct(halluc_pass, len(halluc)),
        "latency_mean": round(statistics.mean(latencies), 2) if latencies else float("nan"),
        "latency_p50": round(pctl(50), 2),
        "latency_p95": round(pctl(95), 2),
        "hops_mean": round(statistics.mean(hops), 2) if hops else float("nan"),
        # Defense-in-depth: how often the model emitted an ungrounded SMILES that
        # the guard had to strip. Scored answers are RAW, so this is reported, not
        # hidden — a non-zero count means the rubric still saw (and judged) the
        # model's hallucination while the live API would have caught it.
        "guard_interventions": sum(1 for r in records if r.get("guard_removed")),
    }


def _format_report(records, agg, tag):
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "=" * 78,
        "AGENT EVALUATION REPORT",
        "=" * 78,
        f"Model:    {MANAGER_MODEL}" + (f"   (tag: {tag})" if tag else ""),
        f"Date:     {when}",
        f"Questions:{agg['n_total']}",
        "-" * 78,
        f"{'ID':<26}{'Category':<14}{'Verdict':<9}{'Lat(s)':>8}{'Hops':>6}",
        "-" * 78,
    ]
    for r in records:
        lines.append(
            f"{r['id']:<26}{r['category']:<14}{r['verdict']:<9}"
            f"{r['latency_s']:>8}{r['llm_hops']:>6}"
        )
    lines += [
        "-" * 78,
        "AGGREGATES",
        "-" * 78,
        f"Automated pass rate:        {agg['auto_pass']}/{agg['n_auto']}  ({agg['auto_pass_rate']})",
        f"Hallucination-probe pass:   {agg['halluc_pass']}/{agg['halluc_total']}  ({agg['halluc_pass_rate']})",
        f"Errors:                     {agg['n_errors']}",
        f"Guard interventions:        {agg['guard_interventions']}  (ungrounded SMILES the live guard would strip; scores above are on RAW model output)",
        f"Latency mean / p50 / p95:   {agg['latency_mean']}s / {agg['latency_p50']}s / {agg['latency_p95']}s",
        f"Mean LLM hops per question: {agg['hops_mean']}",
        "=" * 78,
        "Notes: REVIEW = manual-review item (auto-checks ran but confirm by hand);",
        "REVIEW* = manual-review item that also failed an automated check.",
        "=" * 78,
    ]
    return "\n".join(lines)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"  # x!=x catches nan


def _aggregate_multirun(passes: list[list[dict]]) -> dict:
    """Aggregate N passes: pooled rates with CIs + per-question stability.

    Pooling treats each (question, run) as one trial for the proportion CIs, and
    pools all latencies across runs. Per-question we also record how many runs
    passed, which surfaces FLAKY questions a single run would hide.
    """
    runs = len(passes)
    ids = [r["id"] for r in passes[0]]
    by_id = {rid: [p[i] for p in passes] for i, rid in enumerate(ids)}

    per_question = []
    for rid in ids:
        recs = by_id[rid]
        ok = [r for r in recs if not r["error"]]
        manual = recs[0]["manual_review"]
        n_pass = sum(r["passed"] for r in recs)
        lat = [r["latency_s"] for r in ok]
        if manual:
            verdict = "REVIEW" if n_pass == runs else f"REVIEW {n_pass}/{runs}"
        elif n_pass == runs:
            verdict = "PASS"
        elif n_pass == 0:
            verdict = "FAIL"
        else:
            verdict = f"FLAKY {n_pass}/{runs}"
        per_question.append({
            "id": rid,
            "category": recs[0]["category"],
            "manual_review": manual,
            "pass_count": n_pass,
            "runs": runs,
            "verdict": verdict,
            "latency_mean": round(statistics.mean(lat), 2) if lat else float("nan"),
            "n_errors": runs - len(ok),
        })

    # Pooled trials across all runs for the proportion CIs.
    auto_recs = [r for p in passes for r in p if not r["manual_review"] and not r["error"]]
    halluc_recs = [r for p in passes for r in p if r["category"] == "hallucination" and not r["error"]]
    auto_k = sum(r["passed"] for r in auto_recs)
    halluc_k = sum(r["passed"] for r in halluc_recs)

    latencies = [r["latency_s"] for p in passes for r in p if not r["error"]]
    hops = [r["llm_hops"] for p in passes for r in p if not r["error"]]
    lat_sorted = sorted(latencies)

    def pctl(q):
        if not lat_sorted:
            return float("nan")
        k = max(0, min(len(lat_sorted) - 1, int(round((q / 100) * (len(lat_sorted) - 1)))))
        return lat_sorted[k]

    # Per-run auto pass rate, so we can report run-to-run spread directly.
    per_run_auto_rate = [
        statistics.mean([r["passed"] for r in p if not r["manual_review"] and not r["error"]])
        for p in passes
    ]
    lat_mean, lat_half = mean_ci(latencies)
    auto_lo, auto_hi = wilson_ci(auto_k, len(auto_recs))
    halluc_lo, halluc_hi = wilson_ci(halluc_k, len(halluc_recs))
    # Spread of the per-run pass rate. A standard deviation (not a t-CI) because
    # with the small N of runs here a t-interval blows up and reads as nonsense.
    rate_mean = statistics.mean(per_run_auto_rate) if per_run_auto_rate else float("nan")
    rate_std = statistics.stdev(per_run_auto_rate) if len(per_run_auto_rate) > 1 else float("nan")

    flaky = [q["id"] for q in per_question if q["verdict"].startswith("FLAKY")]

    return {
        "runs": runs,
        "n_questions": len(ids),
        "auto_trials": len(auto_recs),
        "auto_pass": auto_k,
        "auto_pass_rate": auto_k / len(auto_recs) if auto_recs else float("nan"),
        "auto_pass_ci": [auto_lo, auto_hi],
        "auto_rate_per_run_mean": rate_mean,
        "auto_rate_per_run_std": rate_std,
        "halluc_trials": len(halluc_recs),
        "halluc_pass": halluc_k,
        "halluc_pass_rate": halluc_k / len(halluc_recs) if halluc_recs else float("nan"),
        "halluc_pass_ci": [halluc_lo, halluc_hi],
        "flaky_ids": flaky,
        "n_errors": sum(q["n_errors"] for q in per_question),
        "guard_interventions": sum(1 for p in passes for r in p if r.get("guard_removed")),
        "latency_mean": lat_mean,
        "latency_ci_halfwidth": lat_half,
        "latency_p50": round(pctl(50), 2),
        "latency_p95": round(pctl(95), 2),
        "hops_mean": round(statistics.mean(hops), 2) if hops else float("nan"),
        "per_question": per_question,
    }


def _format_multirun_report(agg: dict, tag: str | None) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    runs = agg["runs"]
    lines = [
        "=" * 78,
        f"AGENT EVALUATION REPORT  ({runs} runs)",
        "=" * 78,
        f"Model:    {MANAGER_MODEL}" + (f"   (tag: {tag})" if tag else ""),
        f"Date:     {when}",
        f"Questions:{agg['n_questions']}   Runs: {runs}",
        "-" * 78,
        f"{'ID':<26}{'Category':<14}{'Pass':>7}{'MeanLat':>9}  Verdict",
        "-" * 78,
    ]
    for q in agg["per_question"]:
        lines.append(
            f"{q['id']:<26}{q['category']:<14}"
            f"{str(q['pass_count']) + '/' + str(q['runs']):>7}"
            f"{q['latency_mean']:>9}  {q['verdict']}"
        )
    lo, hi = agg["auto_pass_ci"]
    hlo, hhi = agg["halluc_pass_ci"]
    lat_half = agg["latency_ci_halfwidth"]
    lat_ci = "" if lat_half != lat_half else f"  ±{lat_half:.1f}"
    rate_std = agg["auto_rate_per_run_std"]
    rate_spread = "" if rate_std != rate_std else f"  (±{100 * rate_std:.1f} SD across runs)"
    lines += [
        "-" * 78,
        "AGGREGATES  (pooled over all runs; 95% confidence intervals)",
        "-" * 78,
        f"Automated pass rate:        {agg['auto_pass']}/{agg['auto_trials']}  "
        f"({_pct(agg['auto_pass_rate'])})  [Wilson 95% CI {_pct(lo)}–{_pct(hi)}]",
        f"  per-run mean:             {_pct(agg['auto_rate_per_run_mean'])}{rate_spread}",
        f"Hallucination-probe pass:   {agg['halluc_pass']}/{agg['halluc_trials']}  "
        f"({_pct(agg['halluc_pass_rate'])})  [Wilson 95% CI {_pct(hlo)}–{_pct(hhi)}]",
        f"Flaky auto questions:       {len(agg['flaky_ids'])}"
        + (f"  ({', '.join(agg['flaky_ids'])})" if agg["flaky_ids"] else ""),
        f"Errors:                     {agg['n_errors']}",
        f"Guard interventions:        {agg['guard_interventions']}  (ungrounded SMILES stripped; scores are on RAW output)",
        f"Latency mean:               {agg['latency_mean']:.2f}s{lat_ci} (95% CI)",
        f"Latency p50 / p95:          {agg['latency_p50']}s / {agg['latency_p95']}s",
        f"Mean LLM hops per question: {agg['hops_mean']}",
        "=" * 78,
        "Notes: Pass column = runs passed / total runs. FLAKY = passed some but not",
        "all runs (instability a single run hides). REVIEW = manual-review item.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the agent evaluation suite.")
    parser.add_argument("--tag", help="Label for this run (e.g. '7b', 'multiagent'), used in output filenames.")
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat the whole suite N times and report metrics with 95%% confidence intervals (default 1).")
    parser.add_argument("--dataset", default=None,
                        help="Path to a dataset JSON file (default: eval/dataset.json).")
    parser.add_argument("--html", action="store_true",
                        help="Also generate an HTML report alongside the text report.")
    parser.add_argument("--judge", action="store_true",
                        help="After the deterministic rubric, run the LLM-judge V&V tier "
                             "(eval/judge.py) over the FAILs and rescue answers that are "
                             "correct but did not match the literal rubric (verdict PASS-JUDGE). "
                             "Requires per-question `reference` fields in the dataset.")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    # Allow swapping in an alternate question bank without touching the default.
    if args.dataset:
        global DATASET_PATH
        DATASET_PATH = os.path.abspath(args.dataset)

    passes = run_suite(args.tag, args.runs)

    suffix = f"_{args.tag}" if args.tag else ""
    report_path = os.path.join(HERE, f"eval_report{suffix}.txt")
    json_path = os.path.join(HERE, f"eval_results{suffix}.json")

    if args.runs == 1:
        # Single-run output is the regression-guard format the thesis already uses.
        records = passes[0]

        # Tier-2 V&V: rescue correct-but-non-matching FAILs with the LLM judge.
        # Runs only in single-run mode and only over deterministic FAILs, so it can
        # only raise the auto pass rate, never lower it (see eval/judge.py).
        vv_summary = None
        if args.judge:
            from eval.judge import rejudge_records
            with open(DATASET_PATH) as f:
                questions_by_id = {q["id"]: q for q in json.load(f)["questions"]}
            print("\nRunning LLM-judge V&V tier over deterministic FAILs ...")
            vv_summary = rejudge_records(records, questions_by_id)
            if "error" in vv_summary:
                print(f"  judge skipped: {vv_summary['error']}")
            else:
                print(f"  considered {vv_summary['considered']} FAILs -> "
                      f"rescued {vv_summary['rescued']} (PASS-JUDGE), "
                      f"upheld {vv_summary['upheld']}, abstained {vv_summary['abstained']}")

        agg = _aggregate(records)
        report = _format_report(records, agg, args.tag)
        if vv_summary and "error" not in vv_summary:
            report += (
                f"\n\nV&V (LLM-judge tier): rescued {vv_summary['rescued']} of "
                f"{vv_summary['considered']} deterministic FAILs (upheld "
                f"{vv_summary['upheld']}, abstained {vv_summary['abstained']}). "
                f"PASS-JUDGE items are correct answers the literal rubric missed.\n"
            )
        payload = {"model": MANAGER_MODEL, "tag": args.tag, "aggregates": agg,
                   "vv": vv_summary, "records": records}
    else:
        agg = _aggregate_multirun(passes)
        report = _format_multirun_report(agg, args.tag)
        payload = {"model": MANAGER_MODEL, "tag": args.tag, "runs": args.runs,
                   "aggregates": agg, "passes": passes}

    with open(report_path, "w") as f:
        f.write(report + "\n")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + report)
    print(f"\nReport written to {report_path}")
    print(f"Raw results written to {json_path}")

    if args.html:
        from eval.report_html import generate_html
        html_path = report_path.replace(".txt", ".html")
        records_for_html = passes[0] if args.runs == 1 else [
            {**q, "latency_s": q["latency_mean"], "llm_hops": "-",
             "verdict": q["verdict"], "failures": [], "answer": "", "tools_called": []}
            for q in agg["per_question"]
        ]
        with open(html_path, "w") as f:
            f.write(generate_html(records_for_html, agg, MANAGER_MODEL, args.tag))
        print(f"HTML report written to {html_path}")


if __name__ == "__main__":
    main()

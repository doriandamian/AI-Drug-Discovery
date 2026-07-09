# Agent Evaluation Harness

End-to-end evaluation of the drug-discovery agent. This is **Phase 0** of the
multi-agent roadmap: it establishes the *baseline* the rest of the work is
measured against (single-agent vs multi-agent, model-size ablation, the
generative agent). Every architecture change should be re-run here so the thesis
can show **what changed and by how much**, with numbers.

## What it measures

For each question the harness runs the live LangGraph orchestrator and captures:

| Signal | Why it matters for the thesis |
|---|---|
| **Correctness** (rubric pass/fail) | Does the agent answer with the right facts / caveats? |
| **Tool routing** | Did it call the right tools? (the core of "does multi-agent route better?") |
| **LLM hops** | Number of manager round-trips — the real driver of latency |
| **Latency** (mean / p50 / p95) | The accuracy↔speed trade-off you argue about for model size |
| **Hallucination-probe pass rate** | Does it refuse to invent SMILES / mechanisms / compounds? |

## How to run

Requires the same services as the app: **Ollama** (with the chosen model pulled)
and **Neo4j** running with the RAG index built. Run from `backend/`:

```bash
# Baseline (current single-agent system, model from MANAGER_MODEL, default qwen2.5:14b)
python -m eval.run_eval

# Model-size ablation — same suite, different manager model
MANAGER_MODEL=qwen2.5:7b  python -m eval.run_eval --tag 7b
MANAGER_MODEL=qwen2.5:14b python -m eval.run_eval --tag 14b

# After building the multi-agent system, tag the run so files don't overwrite
python -m eval.run_eval --tag multiagent
```

Outputs (in this folder, suffixed with `--tag` if given):

- `eval_report.txt` — human-readable table + aggregates (drop straight into the thesis)
- `eval_results.json` — raw per-question records (answers, tools, failures) for plots / deeper analysis

To run the larger 300-question pharma bank instead of the default set:

```bash
python -m eval.run_eval --dataset eval/dataset_pharma_300.json --tag pharma300 --judge
```

## Two-tier Verification & Validation (V&V)

The substring/regex rubric is a strict **regression guard**: a PASS is
unambiguous and free. Its weakness is *false FAILs* — an answer that is factually
correct but phrased differently than the rubric anticipated ("~180.2 daltons"
when the rubric wanted "180.1", a refusal worded a new way, targets listed in a
different order). The V&V layer fixes that without weakening the guard:

| Tier | Mechanism | Verdict |
|---|---|---|
| **1 — Verification** | deterministic rubric (`checks.py`) | `PASS` / `FAIL` |
| **2 — Validation** | LLM judge (`judge.py`) over the FAILs only | rescues → `PASS-JUDGE` |

Key properties (see `judge.py`):

- **Monotonic.** The judge only ever turns `FAIL → PASS-JUDGE`, never the reverse
  (a Tier-1 PASS is already trusted and is never re-judged), so it *cannot* lower
  the regression-guard pass rate — worst case it abstains and the FAIL stands.
- **Grounded on a per-question `reference`.** Each dataset question carries a
  natural-language `reference` describing what a valid answer must *convey*. The
  judge grades **meaning, not wording**: it accepts different phrasing / numeric
  precision / ordering, but still rejects fabricated citations, invented
  IC50/LD50/SMILES, missing tool grounding, or dropped safety caveats
  (category-specific guardrails baked into the judge prompt).
- **Offline-safe.** With no Ollama reachable the judge abstains and the harness
  runs the deterministic tier exactly as before.

Run it two ways:

```bash
# During a run — fold V&V into a single-run eval:
python -m eval.run_eval --dataset eval/dataset_pharma_300.json --tag pharma300 --judge

# After the fact — re-score an existing results file, rescuing correct FAILs:
python -m eval.judge --results eval/eval_results_pharma300.json
```

`JUDGE_MODEL` overrides the judge model (default = `MANAGER_MODEL`), so the judge
can be kept independent of the system under test in the size ablation.

## The dataset

Two question banks live here:

- `dataset.json` / `dataset_pharma_100.json` — the original suites.
- `dataset_pharma_300.json` — **300** questions generated reproducibly by
  `gen_dataset_300.py` (seeded, so `python -m eval.gen_dataset_300` reproduces it
  byte-for-byte). Every question has both a `checks` rubric **and** a `reference`
  for the V&V judge. Distribution: properties 75, toxicity 55, literature 50,
  graph 35, design 30, hallucination 55.

Each question carries a per-question **rubric** of automated checks. Categories:

- **properties** — deterministic facts (MW, Lipinski, QED) with checkable values
- **toxicity** — must report endpoints *and* carry the in-vitro / model-scope caveats
- **literature** — evidence retrieval and grounding
- **graph** — cross-compound / relationship questions (targets, diseases, shared endpoints)
- **design** — generative analog design; must use tool structures + unvalidated caveat
- **hallucination** — must refuse to design molecules, must not invent compounds or mechanisms

Check types (see `checks.py`): `must_include`, `must_include_any*` (numbered
OR-groups), `must_not_include`, `expected_tools`, `expected_tools_any`,
`forbidden_tools`, `regex_must`, `regex_must_not`. All matching is
case-insensitive.

Items marked `manual_review: true` still run and report tools/latency, but their
correctness (e.g. quality of literature grounding) is hard to score by substring
and should be confirmed by hand. They show as `REVIEW` / `REVIEW*` in the report.

Alongside `checks`, each question in `dataset_pharma_300.json` has a `reference`
string — a plain-language statement of what a valid answer must convey. This is
what the Tier-2 judge grades against; write it as *acceptance criteria*, not a
single gold string (state tolerances, what counts as valid rephrasing, and what
must NOT appear).

### Adding questions

Append to the dataset (or edit `gen_dataset_300.py` and regenerate). Keep the
`checks` rubric **lenient on wording, strict on what matters** (which tools ran,
required facts/caveats present, no invented data), and add a `reference` so the
judge can validate correct answers the substring rubric would miss.

## Methodological notes (important for the write-up)

- **Shared graph state.** The Neo4j knowledge graph accumulates across questions
  *and across runs* (it self-populates from tool calls). So questions are **not
  independent** — e.g. a cross-compound graph question benefits from earlier
  questions having analysed those compounds. This is realistic (it mirrors a real
  session) but means **run order matters**; keep the dataset order fixed when
  comparing runs, and mention this in the methodology section.
- **Warmup.** The model is loaded into Ollama before timing starts, so the first
  question's latency is not inflated by a one-off cold model load.
- **Single instance = serial.** On local Ollama with one model, "parallel" tool
  or agent calls are serialised. Latency tracks total sequential LLM hops, which
  is exactly why `hops_mean` is reported alongside wall-clock time.

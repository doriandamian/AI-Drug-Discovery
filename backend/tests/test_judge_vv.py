import json
import os

import pytest

from eval.checks import evaluate
from eval.judge import judge_answer, rejudge_records, _extract_json, _coerce

DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "eval", "dataset_pharma_300.json")


class _Resp:
    def __init__(self, content):
        self.content = content


class _StubLLM:
    def __init__(self, content):
        self.content = content
        self.last = None

    def invoke(self, messages):
        self.last = messages
        return _Resp(self.content)


def test_extract_json_tolerates_prose_wrapping():
    assert _coerce(_extract_json('{"valid": true, "reason": "ok"}'))["valid"] is True
    assert _coerce(_extract_json('Sure: {"valid": false, "reason": "no"} .'))["valid"] is False
    assert _coerce(_extract_json('{"valid": "yes"}'))["valid"] is True
    assert _coerce(_extract_json("not json at all")) is None


def test_judge_rescues_correct_but_nonmatching_answer():
    v = judge_answer(
        question="What is the molecular weight of aspirin?",
        reference="Reports ~180.16 g/mol, tool-computed; accept +/-1.",
        answer="Aspirin weighs about 180.2 daltons.",
        tools_called=["calculate_properties"],
        category="properties",
        llm=_StubLLM('{"valid": true, "confidence": 0.95, "reason": "within tolerance"}'),
    )
    assert v["decision"] == "valid"


def test_judge_upholds_wrong_answer():
    v = judge_answer(
        question="MW of aspirin?", reference="~180 g/mol",
        answer="I have no idea.", tools_called=[], category="properties",
        llm=_StubLLM('{"valid": false, "reason": "no value given"}'),
    )
    assert v["decision"] == "invalid"


def test_judge_abstains_without_llm_available(monkeypatch):
    import eval.judge as judge_mod
    monkeypatch.setattr(judge_mod, "get_judge_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ollama")))
    v = judge_answer("q", "some reference", "some answer", category="properties")
    assert v["decision"] == "abstain"


def test_empty_answer_is_invalid_without_calling_llm():
    v = judge_answer("q", "ref", "", category="properties", llm=_StubLLM("unused"))
    assert v["decision"] == "invalid"


def test_category_guardrail_reaches_the_prompt():
    stub = _StubLLM('{"valid": true, "reason": "ok"}')
    judge_answer("q", "ref", "answer", category="hallucination", llm=stub)
    prompt_text = " ".join(m[1] for m in stub.last)
    assert "HALLUCINATION probe" in prompt_text


def test_rejudge_is_monotonic_and_skips_pass_and_manual():
    records = [
        {"id": "a", "passed": False, "verdict": "FAIL", "question": "MW?",
         "answer": "about 180.2 daltons", "category": "properties",
         "tools_called": ["calculate_properties"]},
        {"id": "b", "passed": True, "verdict": "PASS", "question": "x",
         "answer": "y", "category": "properties", "tools_called": []},
        {"id": "c", "passed": False, "manual_review": True, "question": "x",
         "answer": "y", "category": "graph", "tools_called": []},
        {"id": "d", "passed": False, "error": "boom", "question": "x",
         "answer": "", "category": "graph", "tools_called": []},
    ]
    qbi = {r["id"]: {"reference": "~180"} for r in records}
    summary = rejudge_records(records, qbi,
                              llm=_StubLLM('{"valid": true, "reason": "ok"}'))

    assert records[0]["verdict"] == "PASS-JUDGE" and records[0]["passed"] is True
    assert records[1]["verdict"] == "PASS"
    assert "judge" not in records[2]
    assert "judge" not in records[3]
    assert summary == {"considered": 1, "rescued": 1, "upheld": 0, "abstained": 0}


def test_dataset_is_wellformed():
    with open(DATASET) as f:
        data = json.load(f)
    qs = data["questions"]
    assert len(qs) == 300
    ids = [q["id"] for q in qs]
    assert len(set(ids)) == 300, "ids must be unique"
    for q in qs:
        assert q.get("question", "").strip()
        assert isinstance(q.get("checks"), dict)
        assert q.get("reference", "").strip(), f"{q['id']} missing judge reference"
        assert q["category"] in {"properties", "toxicity", "literature",
                                 "graph", "design", "hallucination"}


def test_dataset_checks_run_through_the_rubric_engine():
    with open(DATASET) as f:
        qs = json.load(f)["questions"]
    for q in qs:
        passed, failures = evaluate(q["checks"], "placeholder answer", [])
        assert isinstance(passed, bool) and isinstance(failures, list)

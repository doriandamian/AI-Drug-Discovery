from eval.checks import evaluate
from agents import smiles_guard

HALLUC_CHECK = {"regex_must_not": ["<smiles>"]}
UNGROUNDED = "Sure, here is one: <smiles>CC(C)Cc1ccc(C(C)C(=O)O)cc1</smiles>"


def test_raw_output_with_ungrounded_smiles_fails_the_probe():
    passed, failures = evaluate(HALLUC_CHECK, UNGROUNDED, [])
    assert not passed and failures


def test_sanitising_before_scoring_would_have_masked_it():
    smiles_guard.reset()
    sanitized, removed = smiles_guard.sanitize(UNGROUNDED)
    smiles_guard.reset()
    assert removed, "guard should have stripped the ungrounded SMILES"
    passed_if_masked, _ = evaluate(HALLUC_CHECK, sanitized, [])
    assert passed_if_masked, "sanitised text dodges the probe, exactly why we score RAW"


def test_run_question_keeps_raw_answer_separate_from_delivered():
    import inspect
    from eval import run_eval
    src = inspect.getsource(run_eval._run_question)
    assert '"answer": final' in src, "scored answer must be the RAW model output"
    assert "guard_removed" in src, "guard activity must be recorded separately"

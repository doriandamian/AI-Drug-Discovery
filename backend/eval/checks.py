"""Rubric engine for the evaluation suite.

A question's `checks` block is a small declarative rubric. Each supported key is
an independent assertion; a question PASSES only if every assertion holds. The
checks are deliberately lenient on phrasing (case-insensitive substring / regex)
and strict on the things that actually matter for this agent:

  must_include          [str]  every substring must appear in the answer
  must_include_any*     [str]  at least one must appear. Any key whose name
                               starts with "must_include_any" is its own OR-group
                               (use must_include_any, must_include_any_2, ... to
                               require several independent "at least one" groups).
  must_not_include      [str]  none of these substrings may appear
  expected_tools        [str]  every tool must have been called
  expected_tools_any    [str]  at least one of these tools must have been called
  forbidden_tools       [str]  none of these tools may have been called
  regex_must            [str]  every pattern must match (re.search, IGNORECASE)
  regex_must_not        [str]  no pattern may match

All string matching is case-insensitive. Tool checks read the list of tool names
the agent actually invoked.
"""
import re


def _lower(s: str) -> str:
    return (s or "").lower()


def evaluate(checks: dict, answer: str, tools_called: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, failures). `failures` lists each unmet assertion."""
    failures: list[str] = []
    text = _lower(answer)
    tools = [_lower(t) for t in tools_called]

    for key, value in (checks or {}).items():
        if key == "must_include":
            for sub in value:
                if _lower(sub) not in text:
                    failures.append(f"missing required text: {sub!r}")

        elif key.startswith("must_include_any"):
            if not any(_lower(sub) in text for sub in value):
                failures.append(f"none of the alternatives present: {value}")

        elif key == "must_not_include":
            for sub in value:
                if sub and _lower(sub) in text:
                    failures.append(f"forbidden text present: {sub!r}")

        elif key == "expected_tools":
            for t in value:
                if _lower(t) not in tools:
                    failures.append(f"expected tool not called: {t}")

        elif key == "expected_tools_any":
            if not any(_lower(t) in tools for t in value):
                failures.append(f"none of the expected tools called: {value}")

        elif key == "forbidden_tools":
            for t in value:
                if _lower(t) in tools:
                    failures.append(f"forbidden tool called: {t}")

        elif key == "regex_must":
            for pat in value:
                if not re.search(pat, answer or "", re.IGNORECASE):
                    failures.append(f"required pattern not matched: {pat!r}")

        elif key == "regex_must_not":
            for pat in value:
                if re.search(pat, answer or "", re.IGNORECASE):
                    failures.append(f"forbidden pattern matched: {pat!r}")

    return (len(failures) == 0, failures)

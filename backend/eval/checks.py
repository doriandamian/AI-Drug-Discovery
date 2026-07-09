import re


def _lower(s: str) -> str:
    return (s or "").lower()


def evaluate(checks: dict, answer: str, tools_called: list[str]) -> tuple[bool, list[str]]:
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

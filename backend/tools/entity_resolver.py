import difflib
import re

from core.database import run_read_query

__all__ = ["expand_abbreviations", "stored_entity_names", "fuzzy_resolve"]

ABBREVIATIONS: dict[str, str] = {
    "cox": "cyclooxygenase",
    "cox-1": "cyclooxygenase",
    "cox1": "cyclooxygenase",
    "cox-2": "cyclooxygenase",
    "cox2": "cyclooxygenase",
    "ptgs": "cyclooxygenase",
    "ptgs1": "cyclooxygenase",
    "ptgs2": "cyclooxygenase",

    "ace": "angiotensin-converting enzyme",
    "gaba": "gamma-aminobutyric acid",
    "nmda": "n-methyl-d-aspartate",
    "egfr": "epidermal growth factor receptor",
    "vegf": "vascular endothelial growth factor",
    "tnf": "tumor necrosis factor",
    "hmg-coa": "hydroxymethylglutaryl-coa reductase",
}

_LITERAL_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")

_ABBREV_RE = re.compile(
    r"(?<![\w-])("
    + "|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True))
    + r")(?![\w-])",
    re.IGNORECASE,
)


def _expand_literal(content: str) -> str:
    return _ABBREV_RE.sub(lambda m: ABBREVIATIONS[m.group(1).lower()], content)


def expand_abbreviations(cypher: str) -> tuple[str, bool]:
    changed = False

    def _repl(m: re.Match) -> str:
        nonlocal changed
        new_content = _expand_literal(m.group(1))
        if new_content != m.group(1):
            changed = True
        return "'" + new_content + "'"

    return _LITERAL_RE.sub(_repl, cypher), changed


def stored_entity_names() -> list[str]:
    rows = run_read_query(
        "MATCH (n) WHERE n:Protein OR n:Disease "
        "RETURN DISTINCT n.name AS name LIMIT 1000"
    )
    return [r["name"] for r in rows if r.get("name")]


def fuzzy_resolve(
    cypher: str,
    names: list[str] | None = None,
    cutoff: float = 0.82,
) -> tuple[str, list[tuple[str, str]]]:
    if names is None:
        names = stored_entity_names()
    if not names:
        return cypher, []

    lower_map: dict[str, str] = {}
    for n in names:
        lower_map.setdefault(n.lower(), n)
    candidates = list(lower_map.keys())

    subs: list[tuple[str, str]] = []

    def _repl(m: re.Match) -> str:
        content = m.group(1)
        key = content.strip().lower()
        if not key or key in lower_map:
            return m.group(0)  # empty or already an exact stored name
        close = difflib.get_close_matches(key, candidates, n=1, cutoff=cutoff)
        if not close:
            return m.group(0)
        subs.append((content, lower_map[close[0]]))
        return "'" + close[0] + "'"

    return _LITERAL_RE.sub(_repl, cypher), subs

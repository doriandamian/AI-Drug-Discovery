import re
from contextvars import ContextVar

from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

WITHHELD_MARKER = "[structure withheld: not returned by any tool]"

_provenance: ContextVar[set] = ContextVar("smiles_provenance", default=set())

_TAG_RE = re.compile(r"<smiles>(.*?)</smiles>", re.IGNORECASE | re.DOTALL)
_KV_RE = re.compile(r"smiles\s*=\s*([^\s,;]+)", re.IGNORECASE)
_JSON_RE = re.compile(r'"smiles"\s*:\s*"([^"]+)"', re.IGNORECASE)
_QUOTED_RE = re.compile(r"'([^']+)'")


def reset() -> None:
    _provenance.set(set())


def _canonical(s: str) -> str | None:
    """RDKit-canonical SMILES, or None if it is not a real (≥2-atom) structure."""
    s = (s or "").strip().strip("'\"`")
    if len(s) < 2:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    return Chem.MolToSmiles(mol)


def _add(s: str) -> None:
    canon = _canonical(s)
    if canon:
        _provenance.get().add(canon)


def record_from_text(text: str) -> None:
    """Record any SMILES found in a trustworthy TOOL output.
    """
    if not text:
        return
    for m in _TAG_RE.findall(text):
        _add(m)
    for m in _KV_RE.findall(text):
        _add(m)
    for m in _JSON_RE.findall(text):
        _add(m)


def record_user_message(text: str) -> None:
    """Record SMILES the user provided, whether wrapped in <smiles>...</smiles>
    tags or pasted as a raw structure. A structure the user typed themselves is by
    definition not a hallucination, so the assistant may echo it back verbatim. The
    RDKit validity gate in _canonical keeps ordinary words from being recorded.
    """
    if not text:
        return
    for m in _TAG_RE.findall(text):
        _add(m)
    for token in text.split():
        _add(token.strip(".,;:!?\"'()"))


def sanitize(answer: str) -> tuple[str, list[str]]:
    """Strip ungrounded SMILES from an answer.
    """
    if not answer:
        return answer, []
    prov = _provenance.get()
    removed: list[str] = []

    def _repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        canon = _canonical(inner)
        if canon and canon in prov:
            return m.group(0)
        removed.append(inner)
        return WITHHELD_MARKER

    return _TAG_RE.sub(_repl, answer), removed

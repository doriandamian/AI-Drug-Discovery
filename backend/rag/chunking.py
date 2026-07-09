import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_SECTION_RE = re.compile(
    r"^[ \t]*(?:\d{1,2}[.)]?\s+|[IVX]{1,4}[.)]\s+)?"
    r"(abstract|introduction|background|related work|"
    r"materials and methods|materials|methods?|methodology|experimental(?: section)?|"
    r"results(?: and discussion)?|discussion|conclusions?|"
    r"references|bibliography|acknowledge?ments?|"
    r"supplementary(?: materials?| information)?|appendix)"
    r"\b[ \t]*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_DROP_SECTIONS = {
    "references", "bibliography",
    "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement",
    "supplementary", "appendix",
}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def _normalize_section(raw: str) -> str:
    s = raw.strip().lower()
    if s.startswith("material"):
        return "methods"
    if s.startswith("method") or s.startswith("experimental") or s == "methodology":
        return "methods"
    if s.startswith("results"):
        return "results"
    if s.startswith("conclusion"):
        return "conclusion"
    if s.startswith("acknowledge"):
        return "acknowledgments"
    if s.startswith("supplementary"):
        return "supplementary"
    if s.startswith("related"):
        return "background"
    return s


def _concat_pages(pages: list[Document]):
    parts = []
    breakpoints = []
    offset = 0
    for page in pages:
        page_no = page.metadata.get("page")
        breakpoints.append((offset, page_no))
        text = page.page_content or ""
        parts.append(text)
        offset += len(text) + 1
    return "\n".join(parts), breakpoints


def _page_at(offset: int, breakpoints) -> object:
    page_no = breakpoints[0][1] if breakpoints else None
    for start, no in breakpoints:
        if start > offset:
            break
        page_no = no
    return page_no


def _find_sections(text: str):
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return []

    spans = []
    if matches[0].start() > 0:
        spans.append(("body", 0, matches[0].start()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((_normalize_section(m.group(1)), m.start(), end))
    return spans


def _split_paper(pages: list[Document]) -> list[Document]:
    source = pages[0].metadata.get("source")
    full_text, breakpoints = _concat_pages(pages)
    sections = _find_sections(full_text)

    if not sections:
        return _splitter.split_documents(pages)

    out = []
    for section, start, end in sections:
        if section in _DROP_SECTIONS:
            continue
        section_text = full_text[start:end]
        cursor = 0
        for chunk in _splitter.split_text(section_text):
            local = section_text.find(chunk[:40], cursor)
            if local == -1:
                local = cursor
            cursor = local + 1
            out.append(Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "page": _page_at(start + local, breakpoints),
                    "section": section,
                },
            ))
    return out


def split_pdf_documents(pages: list[Document]) -> list[Document]:
    by_source: dict[str, list[Document]] = {}
    for page in pages:
        by_source.setdefault(page.metadata.get("source"), []).append(page)

    chunks = []
    for source, source_pages in by_source.items():
        source_pages.sort(key=lambda d: d.metadata.get("page") or 0)
        chunks.extend(_split_paper(source_pages))

    for c in chunks:
        c.metadata["source_type"] = "pdf"
    return chunks

import json
import time
import logging
import urllib.error
import urllib.request
import urllib.parse
from langchain_core.tools import tool

from core.config import SEMANTIC_SCHOLAR_API_KEY

__all__ = ["search_semantic_scholar"]

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,venue,citationCount,externalIds"
MAX_RESULTS = 3
ABSTRACT_MAX_CHARS = 800

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


def _build_headers() -> dict:
    headers = {"User-Agent": "drug_discovery_agent/1.0"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


def _fetch_papers(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    params = {
        "query": query,
        "limit": str(max_results),
        "fields": FIELDS,
    }
    url = f"{SEMANTIC_SCHOLAR_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=_build_headers())

    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = int(retry_after) if (retry_after and retry_after.isdigit()) else delay
                logger.warning("Semantic Scholar %d (attempt %d/%d), retrying in %ds...", e.code, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                delay *= 2
            else:
                raise
    return []


def _papers_payload(papers: list[dict]) -> str:
    total = len(papers)
    records = []
    for i, paper in enumerate(papers, 1):
        external = paper.get("externalIds") or {}
        abstract = paper.get("abstract") or "No abstract available."
        if len(abstract) > ABSTRACT_MAX_CHARS:
            abstract = abstract[:ABSTRACT_MAX_CHARS] + "..."
        records.append({
            "index": i,
            "total": total,
            "year": paper.get("year", "N/A"),
            "title": paper.get("title", "No title"),
            "paper_id": paper.get("paperId", "N/A"),
            "pmid": external.get("PubMed"),
            "venue": paper.get("venue") or None,
            "citation_count": paper.get("citationCount"),
            "abstract": abstract,
        })
    return json.dumps({"status": "ok", "source": "semantic_scholar", "count": total,
                       "papers": records})


@tool(description="""Searches Semantic Scholar (~200M papers) using semantic similarity. Returns the top 3 most relevant papers, each labelled PAPER N OF 3. You MUST address all 3 papers in your response, do not silently omit any.

Advantages over search_pubmed: broader coverage (preprints, ML/AI/computational chemistry papers), citation counts as a proxy for scientific impact, and semantic rather than keyword ranking.

Results are automatically saved to the local knowledge base for future search_literature queries. Independent, can run in any step alongside other tools.""")
def search_semantic_scholar(query: str) -> str:
    logger.info("Searching Semantic Scholar for query: %s", query)
    try:
        papers = _fetch_papers(query)
        if not papers:
            return json.dumps({"status": "empty", "source": "semantic_scholar", "count": 0,
                               "papers": [], "message": "No results found on Semantic Scholar "
                                                        "for this query."})

        logger.info("Semantic Scholar: retrieved %d papers.", len(papers))
        result = _papers_payload(papers)

        try:
            from rag.ingest import ingest_semantic_scholar_papers
            ingest_semantic_scholar_papers(papers)
        except Exception:
            logger.warning("S2 RAG ingestion skipped", exc_info=True)

        return result

    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning("Semantic Scholar rate-limited after retries.")
            return json.dumps({"status": "error", "source": "semantic_scholar", "message": (
                "Semantic Scholar is currently rate-limited. Use search_pubmed and "
                "search_literature instead, the local knowledge base is pre-loaded with "
                "pharmaceutical R&D content that covers this topic.")})
        logger.warning("Semantic Scholar HTTP error: %s", e)
        return json.dumps({"status": "error", "source": "semantic_scholar",
                           "message": f"Semantic Scholar search failed (HTTP {e.code})."})
    except Exception as e:
        logger.exception("Semantic Scholar error")
        return json.dumps({"status": "error", "source": "semantic_scholar",
                           "message": f"Semantic Scholar search failed: {e}"})

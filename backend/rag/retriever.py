import json
import logging
import threading

from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from flashrank import Ranker, RerankRequest

from core.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    OLLAMA_BASE_URL as OLLAMA_URL,
    VECTOR_INDEX_NAME,
    KEYWORD_INDEX_NAME,
    RERANK_MODEL,
    RERANK_CACHE_DIR,
)

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

CANDIDATE_K = 10
FINAL_N = 3

RERANK_THRESHOLD = 0.05

_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
_vector_store = None
_ranker = None
_init_lock = threading.Lock()


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        with _init_lock:
            if _vector_store is None:
                _vector_store = Neo4jVector.from_existing_index(
                    embedding=_embeddings,
                    url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    index_name=VECTOR_INDEX_NAME,
                    keyword_index_name=KEYWORD_INDEX_NAME,
                    search_type="hybrid",
                    text_node_property="text",
                )
    return _vector_store


def _get_ranker():
    global _ranker
    if _ranker is None:
        with _init_lock:
            if _ranker is None:
                _ranker = Ranker(model_name=RERANK_MODEL, cache_dir=RERANK_CACHE_DIR)
    return _ranker


def warmup():
    _get_vector_store()
    _get_ranker()


def _rerank(query: str, candidates):
    if not candidates:
        return []

    passages = [
        {"id": i, "text": doc.page_content}
        for i, (doc, _) in enumerate(candidates)
    ]
    ranked = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
    return [(candidates[r["id"]][0], float(r["score"])) for r in ranked]


@tool(description="""Searches the LOCAL knowledge base: a Neo4j index built from (a) PDFs manually ingested into the system and (b) PubMed / Semantic Scholar abstracts cached from prior searches. Uses hybrid retrieval, vector (semantic) similarity combined with BM25 keyword matching, so exact terms (gene names, drug names, SMILES) and paraphrases both rank well.

Returns up to 3 chunks. Each result includes the source (PMID or PDF filename + page), title, date, and a relevance score; chunks below the minimum relevance threshold are excluded.

Prefer this tool when:
- The question is a follow-up on a compound or topic already researched in this or a prior session.
- You want to find patterns or cross-compound connections across multiple prior searches.
- You want to avoid a redundant PubMed round-trip when the answer is likely already cached.

Use both search_literature AND search_pubmed together for thorough research.
Independent, can run in any step alongside other tools.""")
def search_literature(query: str):
    try:
        store = _get_vector_store()
        candidates = store.similarity_search_with_score(query, k=CANDIDATE_K)

        reranked = _rerank(query, candidates)
        relevant = [
            (doc, score) for doc, score in reranked
            if score >= RERANK_THRESHOLD
        ][:FINAL_N]

        if not relevant:
            return json.dumps({"status": "empty", "source": "local_kb", "count": 0, "chunks": [],
                               "message": (f"No sufficiently relevant documents found in the local "
                                           f"knowledge base for '{query}' (best score below "
                                           f"{RERANK_THRESHOLD:.2f}). Try search_pubmed to find new "
                                           f"literature.")})

        chunks = []
        for doc, score in relevant:
            chunks.append({
                "source": doc.metadata.get("source", "Unknown"),
                "pmid": doc.metadata.get("pmid"),
                "title": doc.metadata.get("title", ""),
                "date": doc.metadata.get("date", ""),
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section"),
                "relevance_score": round(float(score), 2),
                "text": doc.page_content,
            })
        return json.dumps({"status": "ok", "source": "local_kb", "count": len(chunks),
                           "chunks": chunks})

    except Exception:
        logger.exception("RAG search error")
        return json.dumps({"status": "error", "source": "local_kb",
                           "message": "Local knowledge base is unavailable. Please ingest documents "
                                      "first or check the database connection."})

import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase

from rag.chunking import split_pdf_documents

from core.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    OLLAMA_BASE_URL as OLLAMA_URL,
    VECTOR_INDEX_NAME,
    KEYWORD_INDEX_NAME,
)

logger = logging.getLogger(__name__)


def _embeddings():
    return OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)


def _store_documents(docs: list[Document]):
    Neo4jVector.from_documents(
        documents=docs,
        embedding=_embeddings(),
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=VECTOR_INDEX_NAME,
        keyword_index_name=KEYWORD_INDEX_NAME,
        search_type="hybrid",
        node_label="DocumentChunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )


def ensure_indexes():
    if not get_processed_sources():
        return
    try:
        Neo4jVector.from_existing_graph(
            embedding=_embeddings(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=VECTOR_INDEX_NAME,
            keyword_index_name=KEYWORD_INDEX_NAME,
            search_type="hybrid",
            node_label="DocumentChunk",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        logger.info("RAG: ensured indexes '%s' + '%s' exist.", VECTOR_INDEX_NAME, KEYWORD_INDEX_NAME)
    except Exception:
        logger.exception("RAG: ensure_indexes failed")


def get_processed_sources():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    sources = []
    with driver.session() as session:
        result = session.run("MATCH (d:DocumentChunk) RETURN DISTINCT d.source AS source")
        for record in result:
            if record["source"] is not None:
                sources.append(record["source"])
    driver.close()
    return sources


def get_processed_files():
    return [s for s in get_processed_sources() if not s.startswith("PubMed:PMID:")]


def is_corpus_preloaded() -> bool:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(
            "MATCH (m:CorpusPreloadMarker) RETURN count(m) AS n"
        )
        exists = result.single()["n"] > 0
    driver.close()
    return exists


def mark_corpus_preloaded():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run(
            "MERGE (m:CorpusPreloadMarker) SET m.completed_at = datetime()"
        )
    driver.close()


def ingest_pubmed_articles(articles: list[dict]):
    if not articles:
        return

    existing = set(get_processed_sources())
    new_articles = [
        a for a in articles
        if a.get("pmid") and f"PubMed:PMID:{a['pmid']}" not in existing
    ]

    if not new_articles:
        return

    docs = []
    for article in new_articles:
        text = (
            f"Title: {article.get('title', 'No title')}\n\n"
            f"Abstract: {article.get('abstract', 'No abstract available.')}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": f"PubMed:PMID:{article['pmid']}",
                "source_type": "pubmed",
                "pmid": article["pmid"],
                "title": article.get("title", ""),
                "date": article.get("date", ""),
            }
        ))

    _store_documents(docs)
    logger.info("RAG: ingested %d new PubMed abstract(s) into vector store.", len(docs))

def ingest_semantic_scholar_papers(papers: list[dict]):
    if not papers:
        return

    existing = set(get_processed_sources())
    docs = []

    for paper in papers:
        external = paper.get("externalIds") or {}
        pmid = external.get("PubMed")
        paper_id = paper.get("paperId", "")

        source_key = f"PubMed:PMID:{pmid}" if pmid else f"S2:{paper_id}"
        if source_key in existing:
            continue

        abstract = paper.get("abstract") or "No abstract available."
        title = paper.get("title") or "No title"
        year = str(paper.get("year") or "")

        text = f"Title: {title}\n\nAbstract: {abstract}"

        metadata = {
            "source": source_key,
            "source_type": "semantic_scholar",
            "title": title,
            "date": year,
        }
        if pmid:
            metadata["pmid"] = pmid
        if paper_id:
            metadata["s2_paper_id"] = paper_id
        if paper.get("venue"):
            metadata["venue"] = paper["venue"]

        docs.append(Document(page_content=text, metadata=metadata))
        existing.add(source_key)

    if not docs:
        return

    _store_documents(docs)
    logger.info("RAG: ingested %d new Semantic Scholar paper(s) into vector store.", len(docs))


def build_vector_store():
    processed_files = get_processed_files()
    raw_data_path = "data/raw"

    if not os.path.exists(raw_data_path):
        logger.warning("Raw data directory '%s' does not exist. Skipping vector store build.", raw_data_path)
        return

    all_files = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path) if f.endswith('.pdf')]
    new_files = [f for f in all_files if f not in processed_files]

    if not new_files:
        logger.info("No new files to process. Skipping vector store build.")
        return

    logger.info("Processing %d new files...", len(new_files))

    all_new_docs = []
    for file in new_files:
        logger.info("Loading file: %s", file)
        loader = PyPDFLoader(file)
        docs = loader.load()
        all_new_docs.extend(docs)

    docs = split_pdf_documents(all_new_docs)

    _store_documents(docs)
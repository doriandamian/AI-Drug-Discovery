import json
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from langchain_core.tools import tool

from core.config import NCBI_TOOL, NCBI_EMAIL

__all__ = ["search_pubmed"]

ABSTRACT_MAX_CHARS = 800

logger = logging.getLogger(__name__)

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MAX_RESULTS = 3


def _search_pubmed_pmids(query: str, max_results: int = MAX_RESULTS) -> list[str]:
    """Use esearch to find the most relevant PMIDs for a query."""
    params = {
        "db": "pubmed",
        "term": query,
        "sort": "relevance",
        "retmax": str(max_results),
        "rettype": "json",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    url = f"{PUBMED_ESEARCH_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
        root = ET.fromstring(body)
        pmids = []
        for id_elem in root.findall(".//Id"):
            if id_elem.text:
                pmids.append(id_elem.text)
        return pmids
    except Exception as e:
        raise RuntimeError(f"PubMed esearch failed: {str(e)}")


def _fetch_pubmed_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []

    pmids_str = ",".join(pmids)
    params = {
        "db": "pubmed",
        "id": pmids_str,
        "rettype": "xml",
        "retmode": "xml",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    url = f"{PUBMED_EFETCH_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
        root = ET.fromstring(body)

        articles = []
        for pubmed_article in root.findall(".//PubmedArticle"):
            article_data = {}

            pmid_elem = pubmed_article.find(".//PMID")
            article_data["pmid"] = pmid_elem.text if pmid_elem is not None else "N/A"

            pub_date = pubmed_article.find(".//PubDate")
            if pub_date is not None:
                year = pub_date.find("Year")
                month = pub_date.find("Month")
                day = pub_date.find("Day")
                date_parts = [year.text if year is not None else ""]
                if month is not None:
                    date_parts.append(month.text)
                if day is not None:
                    date_parts.append(day.text)
                article_data["date"] = "-".join(filter(None, date_parts))
            else:
                article_data["date"] = "N/A"

            title_elem = pubmed_article.find(".//ArticleTitle")
            article_data["title"] = title_elem.text if title_elem is not None else "No title"

            abstract_parts = []
            for abstract_elem in pubmed_article.findall(".//AbstractText"):
                text = "".join(abstract_elem.itertext()).strip()
                if not text:
                    continue
                label = abstract_elem.get("Label")
                abstract_parts.append(f"{label}: {text}" if label else text)
            article_data["abstract"] = (
                "\n".join(abstract_parts) if abstract_parts else "No abstract available."
            )

            articles.append(article_data)

        return articles
    except Exception as e:
        raise RuntimeError(f"PubMed efetch failed: {str(e)}")


def _papers_payload(articles: list[dict]) -> str:
    total = len(articles)
    papers = []
    for i, article in enumerate(articles, 1):
        abstract = article.get("abstract", "No abstract available.")
        if len(abstract) > ABSTRACT_MAX_CHARS:
            abstract = abstract[:ABSTRACT_MAX_CHARS] + "..."
        papers.append({
            "index": i,
            "total": total,
            "date": article.get("date", "N/A"),
            "title": article.get("title", "N/A"),
            "pmid": article.get("pmid", "N/A"),
            "abstract": abstract,
        })
    return json.dumps({"status": "ok", "source": "pubmed", "count": total, "papers": papers})


@tool(description="Searches PubMed for scientific article abstracts using relevance ranking. Returns the top 3 most relevant papers (ranked by relevance to the query, not by date), each labelled PAPER N OF 3. You must address all 3 papers in your response. Results are automatically saved to the local knowledge base for future search_literature queries.")
def search_pubmed(query: str) -> str:
    logger.info("Searching PubMed for query: %s", query)
    try:
        pmids = _search_pubmed_pmids(query, max_results=MAX_RESULTS)
        if not pmids:
            return json.dumps({"status": "empty", "source": "pubmed", "count": 0, "papers": [],
                               "message": "No results found for the given query."})

        logger.info("Found %d relevant PMIDs: %s", len(pmids), pmids)

        articles = _fetch_pubmed_abstracts(pmids)
        result = _papers_payload(articles)
        logger.info("PubMed search completed. Retrieved %d articles.", len(articles))

        try:
            from rag.ingest import ingest_pubmed_articles
            ingest_pubmed_articles(articles)
        except Exception:
            logger.warning("RAG ingestion skipped", exc_info=True)

        return result

    except Exception as e:
        logger.exception("An error occurred while searching PubMed")
        return json.dumps({"status": "error", "source": "pubmed",
                           "message": f"An error occurred while searching PubMed: {e}"})

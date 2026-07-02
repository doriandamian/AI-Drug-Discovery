import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pubmed_api import _search_pubmed_pmids, _fetch_pubmed_abstracts
from rag.ingest import ingest_pubmed_articles, get_processed_sources

logger = logging.getLogger(__name__)

PAPERS_PER_TOPIC = 50

RATE_LIMIT_DELAY = 0.4

FETCH_BATCH_SIZE = 50

TOPICS = [
    ("ADMET drug discovery pharmacokinetics",               "ADMET"),
    ("drug-induced liver injury hepatotoxicity mechanism",  "Hepatotoxicity"),
    ("hERG cardiotoxicity QT prolongation drug",            "Cardiotoxicity / hERG"),
    ("Tox21 toxicology assay high-throughput screening",    "Tox21 assays"),
    ("drug toxicity prediction machine learning",           "ML toxicity prediction"),
    ("Lipinski rule of five drug-likeness oral bioavail",   "Drug-likeness / Lipinski"),
    ("CYP450 drug metabolism inhibition induction",         "CYP450 metabolism"),
    ("drug-drug interaction pharmacokinetics",              "Drug-drug interactions"),
    ("blood-brain barrier permeability CNS drug",           "BBB / CNS drugs"),
    ("structure-activity relationship medicinal chemistry", "SAR"),
    ("molecular docking virtual screening drug target",     "Docking / virtual screening"),
    ("lead optimization drug discovery hit",                "Lead optimization"),
    ("drug repurposing repositioning approved drug",        "Drug repurposing"),
    ("clinical trial failure attrition toxicity efficacy",  "Clinical trial failures"),
    ("protein-ligand binding affinity free energy",         "Protein-ligand binding"),
    ("kinase inhibitor selectivity cancer therapy",         "Kinase inhibitors"),
    ("GPCR drug target allosteric modulator",               "GPCR targets"),
    ("nuclear receptor ligand drug endocrine",              "Nuclear receptors"),
    ("natural product drug discovery bioactive compound",   "Natural products"),
    ("antibody drug conjugate ADC cancer",                  "ADCs / biologics"),
    ("antimicrobial resistance antibiotic mechanism",       "Antimicrobial resistance"),
    ("PROTAC targeted protein degradation",                 "PROTACs"),
    ("fragment-based drug discovery FBDD",                  "Fragment-based discovery"),
    ("renal nephrotoxicity drug kidney",                    "Nephrotoxicity"),
    ("neurotoxicity CNS adverse effect drug",               "Neurotoxicity"),
]


def preload():
    logger.info("Pharmaceutical R&D corpus pre-load, %d topics, %d papers/topic", len(TOPICS), PAPERS_PER_TOPIC)

    existing = set(get_processed_sources())
    total_ingested = 0
    total_already_present = 0

    for query, label in TOPICS:
        logger.info("[%s] query: %s", label, query)

        try:
            pmids = _search_pubmed_pmids(query, max_results=PAPERS_PER_TOPIC)
            time.sleep(RATE_LIMIT_DELAY)
        except Exception:
            logger.warning("[%s] esearch failed", label, exc_info=True)
            continue

        new_pmids = [p for p in pmids if f"PubMed:PMID:{p}" not in existing]
        already = len(pmids) - len(new_pmids)
        total_already_present += already

        if not new_pmids:
            logger.info("[%s] all %d papers already in knowledge base. Skipping.", label, len(pmids))
            continue

        logger.info("[%s] %d PMIDs found, %d already stored, %d new.", label, len(pmids), already, len(new_pmids))

        for i in range(0, len(new_pmids), FETCH_BATCH_SIZE):
            batch = new_pmids[i : i + FETCH_BATCH_SIZE]
            try:
                articles = _fetch_pubmed_abstracts(batch)
                ingest_pubmed_articles(articles)
                for a in articles:
                    existing.add(f"PubMed:PMID:{a['pmid']}")
                total_ingested += len(articles)
                batch_num = i // FETCH_BATCH_SIZE + 1
                logger.info("[%s] batch %d: ingested %d abstracts.", label, batch_num, len(articles))
                time.sleep(RATE_LIMIT_DELAY)
            except Exception:
                logger.warning("[%s] batch fetch failed", label, exc_info=True)
                continue

    from rag.ingest import mark_corpus_preloaded
    mark_corpus_preloaded()

    logger.info(
        "Pre-load complete. New abstracts ingested: %d, already in knowledge base: %d",
        total_ingested, total_already_present,
    )


def run_if_needed():
    from rag.ingest import is_corpus_preloaded
    if is_corpus_preloaded():
        logger.info("Corpus preload skipped, preload marker found in knowledge base.")
        return
    logger.info("Corpus preload starting, no preload marker found.")
    preload()


if __name__ == "__main__":
    preload()

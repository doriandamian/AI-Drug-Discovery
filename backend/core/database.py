import re
import time
import logging
import threading
from neo4j import GraphDatabase, exceptions

from core.config import NEO4J_URI as URI, NEO4J_AUTH as AUTH

logger = logging.getLogger(__name__)

_driver = None
_driver_lock = threading.Lock()


def _normalize_name(name: str) -> str:
    return name.strip().capitalize()

def get_driver():
    global _driver
    if _driver is None:
        with _driver_lock:
            if _driver is None:
                _driver = GraphDatabase.driver(URI, auth=AUTH)
    return _driver

def test_connection():
    with get_driver().session() as session:
        result = session.run("RETURN 1 AS number")
        record = result.single()
        logger.info("Neo4j test query returned: %s", record["number"])

def wait_for_neo4j(retries: int = 20, delay: float = 2.0):
    for attempt in range(1, retries + 1):
        try:
            get_driver().verify_connectivity()
            return
        except exceptions.ServiceUnavailable:
            logger.warning("Neo4j not ready (attempt %d/%d), retrying in %.0fs...", attempt, retries, delay)
            time.sleep(delay)
    raise RuntimeError(f"Neo4j unavailable after {retries} attempts ({retries * delay:.0f}s)")

def upsert_compound(name: str, properties: dict[str, object]) -> None:
    query = """
    MERGE (c:Compound {name: $name})
    SET c += $props
    SET c.updated_at = datetime()
    RETURN c
    """
    with get_driver().session() as session:
        session.run(query, name=_normalize_name(name), props=properties)

def get_compound(name: str) -> dict[str, object] | None:
    query = "MATCH (c:Compound {name: $name}) RETURN c LIMIT 1"

    with get_driver().session() as session:
        result = session.run(query, name=_normalize_name(name)).single()
        if result:
            return dict(result['c'])
    return None


_CYPHER_BLOCKLIST = re.compile(
    r'\b(LOAD\s+CSV|apoc\.load|apoc\.import|apoc\.export|apoc\.util\.sleep'
    r'|CALL\s+apoc|dbms\.|PERIODIC\s+COMMIT|USING\s+PERIODIC'
    r'|CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP)\b',
    re.IGNORECASE,
)


def run_read_query(cypher: str, params: dict | None = None, limit: int = 50) -> list[dict]:
    if _CYPHER_BLOCKLIST.search(cypher):
        raise ValueError(
            "Blocked: query contains a disallowed clause (write clauses "
            "CREATE/MERGE/SET/DELETE/DETACH/REMOVE/DROP, LOAD CSV, apoc.load/import/export, dbms.*)"
        )
    with get_driver().session() as session:
        rows = session.execute_read(
            lambda tx: [r.data() for _, r in zip(range(limit), tx.run(cypher, params or {}))]
        )
    return rows


def link_toxicity_endpoints(name: str, results: list[tuple[str, float, float, bool]]) -> None:
    query = """
    MERGE (c:Compound {name: $name})
    WITH c
    UNWIND $rows AS row
    MERGE (e:ToxicityEndpoint {id: row.endpoint})
    MERGE (c)-[r:HAS_TOXICITY]->(e)
    SET r.probability = row.probability,
        r.cutoff = row.cutoff,
        r.flagged = row.flagged,
        r.updated_at = datetime()
    """
    rows = [
        {
            "endpoint": ep,
            "probability": round(float(prob), 4),
            "cutoff": round(float(cutoff), 4),
            "flagged": bool(flagged),
        }
        for ep, prob, cutoff, flagged in results
    ]
    with get_driver().session() as session:
        session.run(query, name=_normalize_name(name), rows=rows)


def link_targets(name: str, targets: list[dict[str, object]]) -> None:
    query = """
    MERGE (c:Compound {name: $name})
    WITH c
    UNWIND $rows AS row
    MERGE (p:Protein {chembl_id: row.chembl_id})
    SET p.name = row.name, p.organism = row.organism
    MERGE (c)-[r:TARGETS]->(p)
    SET r.mechanism = row.mechanism,
        r.action_type = row.action_type,
        r.updated_at = datetime()
    """
    rows = [t for t in targets if t.get("chembl_id")]
    if not rows:
        return
    with get_driver().session() as session:
        session.run(query, name=_normalize_name(name), rows=rows)


def link_indications(name: str, diseases: list[dict[str, object]]) -> None:
    query = """
    MERGE (c:Compound {name: $name})
    WITH c
    UNWIND $rows AS row
    MERGE (d:Disease {name: row.name})
    SET d.mesh_id = row.mesh_id
    MERGE (c)-[r:TREATS]->(d)
    SET r.max_phase = row.max_phase,
        r.updated_at = datetime()
    """
    rows = [d for d in diseases if d.get("name")]
    if not rows:
        return
    with get_driver().session() as session:
        session.run(query, name=_normalize_name(name), rows=rows)
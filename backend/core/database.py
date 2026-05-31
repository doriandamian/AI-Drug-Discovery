import time
from neo4j import GraphDatabase, exceptions

URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "testpassword123")

def test_connection():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run("RETURN 1 AS number")
            record = result.single()
            print(record["number"])

def wait_for_neo4j(retries: int = 20, delay: float = 2.0):
    for attempt in range(1, retries + 1):
        try:
            driver = GraphDatabase.driver(URI, auth=AUTH)
            driver.verify_connectivity()
            return
        except exceptions.ServiceUnavailable:
            print(f"Neo4j not ready (attempt {attempt}/{retries}), retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"Neo4j unavailable after {retries} attempts ({retries * delay:.0f}s)")

def upsert_compound(name, properties):
    query = """
    MERGE (c:Compound {name: $name})
    SET c.smiles = $smiles
    SET c.molecular_weight = $molecular_weight
    SET c.xlogp = $xlogp
    SET c.updated_at = datetime()
    RETURN c
    """

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            session.run(
                query,
                name=name.capitalize(),
                smiles=properties.get('smiles'),
                molecular_weight=properties.get('mw'),
                xlogp=properties.get('logp')
            )

def get_compound(name):
    query = "MATCH (c:Compound {name: $name}) RETURN c LIMIT 1"

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(query, name=name.capitalize()).single()
            if result:
                node = result['c']
                return {
                    'smiles': node.get('smiles'),
                    'molecular_weight': node.get('molecular_weight'),
                    'xlogp': node.get('xlogp')
                }
    return None
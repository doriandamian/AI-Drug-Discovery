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

def wait_for_neo4j():
    for _ in range(20):
        try:
            driver = GraphDatabase.driver(
                "bolt://neo4j:7687",
                auth=("neo4j", "testpassword123")
            )
            driver.verify_connectivity()
            return
        except exceptions.ServiceUnavailable: # Catch specific Neo4j connection error
            time.sleep(2)

def upsert_compound(name, properties):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(URI, auth=AUTH)

    query = """
    MERGE (c:Compound {name: $name})
    SET c.smiles = $smiles
    SET c.molecular_weight = $molecular_weight
    SET c.xlogp = $xlogp
    SET c.updated_at = datetime()
    RETURN c
    """

    with driver.session() as session:
        result = session.run(
            query,
            name=name.capitalize(),
            smiles=properties.get('smiles'),
            molecular_weight=properties.get('molecular_weight'),
            xlogp=properties.get('xlogp')
        )
    driver.close()

def get_compound(name):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(URI, auth=AUTH)

    query = "MATCH (c:Compound {name: $name}) RETURN c"

    with driver.session() as session:
        result = session.run(query, name=name.capitalize()).single()
        if result:
            node = result['c']
            return {
                'smiles': node.get('smiles'),
                'molecular_weight': node.get('molecular_weight'),
                'xlogp': node.get('xlogp')
            }
    driver.close()
    return None
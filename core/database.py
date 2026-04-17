import time
from neo4j import GraphDatabase

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
        except:
            time.sleep(2)
import core.database as db

if __name__ == "__main__":
    print("BACKEND STARTED")

    print("WAITING FOR NEO4J...")
    db.wait_for_neo4j()
    print("CONNECTED TO NEO4J")

    while True:
        pass
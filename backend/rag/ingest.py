import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "testpassword123"

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

def get_processed_files():
    """Returns a set of filenames that have already been processed and stored in Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    processed_files = []
    with driver.session() as session:
        result = session.run("MATCH (d:DocumentChunk) RETURN DISTINCT d.source AS source")
        for record in result:
            if record["source"] is not None:
                processed_files.append(record["source"])
    driver.close()
    return processed_files

def build_vector_store():
    processed_files = get_processed_files()
    raw_data_path = "data/raw"

    if not os.path.exists(raw_data_path):
        print(f"Raw data directory '{raw_data_path}' does not exist. Skipping vector store build.")
        return
    
    all_files = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path) if f.endswith('.pdf')]
    new_files = [f for f in all_files if f not in processed_files]

    if not new_files:
        print("No new files to process. Skipping vector store build.")
        return

    print(f"Processing {len(new_files)} new files...")

    all_new_docs = []
    for file in new_files:
        print(f"Loading file: {file}")
        loader = PyPDFLoader(file)
        docs = loader.load()
        all_new_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(all_new_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)

    Neo4jVector.from_documents(
        documents=docs,
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="local_pdf",
        node_label="DocumentChunk",
        text_node_property="text",
        embedding_node_property="embedding"
    )
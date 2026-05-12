import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "testpassword123"

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

def build_vector_store():
    print("Reading PDF files from data/raw/ ...")
    loader = PyPDFDirectoryLoader("./data/raw")
    documents = loader.load()

    if not documents:
        print("No PDF files found in data/raw/. Please add some PDF files and try again.")
        return
    
    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print("Creating embeddings using Ollama...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )

    print("Storing embeddings in Neo4j...")
    vector_store = Neo4jVector.from_documents(
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

    print("Vector store built and stored in Neo4j successfully.")

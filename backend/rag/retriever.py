import os

from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "testpassword123"

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
_vector_store = None

def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = Neo4jVector.from_existing_index(
            embedding=_embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="local_pdf",
            text_node_property="text",
        )
    return _vector_store

@tool(description="Searches the local Neo4j vector database for private medical literature and PDF documents. Input: A medical question in English.")
def search_literature(query: str):
    try:
        results = _get_vector_store().similarity_search(query, k=3)
        
        formatted_results = []
        for doc in results:
            # Preluăm sursa și pagina dacă există în metadate, altfel afișăm doar textul
            sursa = doc.metadata.get('source', 'Document Necunoscut')
            pagina = doc.metadata.get('page', '?')
            formatted_results.append(f"- Source ({sursa}, Page {pagina}):\n{doc.page_content}\n")
            
        if not formatted_results:
            return "No relevant information found in the local documents."
            
        return "\n".join(formatted_results)
    except Exception as e:
        print(f"RAG Error: {str(e)}")
        return "Local database is empty at the time. Please ingest documents first or check the database connection."
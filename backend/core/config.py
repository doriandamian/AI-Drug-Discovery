import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise RuntimeError(
        "NEO4J_PASSWORD is not set. Copy .env.example to .env and set a real password."
    )
NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

MANAGER_MODEL = os.getenv("MANAGER_MODEL", "qwen2.5:14b")

SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", "qwen2.5:7b")

VECTOR_INDEX_NAME = os.getenv("RAG_VECTOR_INDEX", "knowledge_base")
KEYWORD_INDEX_NAME = os.getenv("RAG_KEYWORD_INDEX", "knowledge_base_keyword")

RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
RERANK_CACHE_DIR = os.getenv(
    "RERANK_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", ".flashrank_cache"),
)

NCBI_TOOL = os.getenv("NCBI_TOOL", "drug_discovery_agent")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "research@example.com")

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

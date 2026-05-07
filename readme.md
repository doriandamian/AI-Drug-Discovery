# 🧬 AI Drug Discovery Assistant

An on-premise multi-agent AI platform for secure and explainable drug discovery research.

The system combines:

- Local LLMs (Ollama)
- Graph databases (Neo4j)
- Cheminformatics tools (RDKit)
- Retrieval-Augmented Generation (RAG)

to provide AI-assisted molecular analysis while keeping proprietary data fully local.

---

# 🚀 Features

- Multi-agent workflow orchestration
- Literature retrieval from local research papers
- Molecular property analysis
- Toxicity and drug-likeness validation
- Graph-based explainability
- Fully local deployment with Docker

---

# 🏗️ Architecture

```text
User → FastAPI Backend → Orchestrator Agent
                         ├── Literature RAG
                         ├── Cheminformatics Agent
                         ├── Safety Agent
                         ├── Neo4j
                         └── Ollama
```

---

# 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Backend | FastAPI, Python |
| AI | Ollama, LangGraph, LangChain |
| Database | Neo4j |
| Cheminformatics | RDKit |
| Deployment | Docker Compose |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/doriandamian/AI-Drug-Discovery

cd AI-Drug-Discovery
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start Services

```bash
docker compose up --build
```

---


# 💡 Example Use Cases

- Drug candidate analysis
- Toxicity screening
- Molecular property prediction
- Literature-assisted discovery
- Research paper semantic search

---

# 🔒 Advantages

- Fully local / on-premise
- Protects proprietary molecular data
- Reduces LLM hallucinations using deterministic tools
- Explainable graph-based reasoning


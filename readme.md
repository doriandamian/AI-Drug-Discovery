# AI Drug Discovery Assistant

A multi-agent research assistant for early-stage drug discovery that runs entirely
on local infrastructure. It answers questions about molecular properties, toxicity,
published evidence, and target-disease relationships, and it can generate novel
analogs of a known compound. Because the language models, the knowledge graph, and
the paper corpus all run on-premise, proprietary molecular data never leaves the
machine.

The system is designed to be grounded rather than generative-by-default: the
language model does not answer chemistry questions from memory. Every factual claim
is produced by a deterministic tool (RDKit, a trained toxicity model, a Cypher query
over Neo4j, or retrieval over the local corpus), and structures are never invented
by the model.

## Architecture

The assistant is a hierarchical multi-agent system. A supervisor model
(`qwen2.5:14b`) receives the user's question and routes it to one or more specialist
sub-agents (`qwen2.5:7b`), each a focused ReAct agent with its own prompt and only
the tools it needs. The supervisor then composes the specialists' results into a
single answer; it holds no chemistry tools of its own.

The specialists share state through a Neo4j knowledge graph that acts as a
blackboard: data one agent writes (toxicity endpoints, drug targets, indications)
becomes queryable by another, so results accumulate across a session.

```text
User → FastAPI (SSE stream) → Supervisor Agent (qwen2.5:14b)
                               │ routes to specialists (qwen2.5:7b)
                               ├── Cheminformatics   → PubChem, RDKit (properties, Lipinski, QED)
                               ├── Safety/Toxicology → multi-task LightGBM toxicity model
                               ├── Literature        → local RAG (hybrid + rerank), PubMed, Semantic Scholar
                               ├── Knowledge Graph   → Neo4j (Cypher), ChEMBL enrichment
                               └── Molecular Design  → BRICS genetic algorithm + multi-objective oracle
                                            ▲
                            shared blackboard: Neo4j knowledge graph
```

The molecular design specialist is the only component that produces new structures.
It fragments a named seed compound with BRICS, recombines the fragments with a
genetic algorithm, and ranks candidates with a multi-objective oracle (drug-likeness,
predicted toxicity, synthetic accessibility, structural alerts, and applicability
domain). Every structure that reaches the user comes verbatim from a tool, guarded by
a deterministic check that strips any SMILES the model did not get from one.

To run on a single model and avoid loading two, set `SUBAGENT_MODEL=qwen2.5:14b` so
the specialists reuse the supervisor's model.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python |
| Orchestration | LangGraph, LangChain |
| Language models | Ollama (qwen2.5:14b / qwen2.5:7b) |
| Embeddings | Ollama (nomic-embed-text) |
| Knowledge graph | Neo4j |
| Cheminformatics | RDKit |
| Toxicity model | LightGBM (multi-task, trained on Tox21 and ClinTox) |
| Deployment | Docker Compose |

## Prerequisites

- Docker and Docker Compose
- Python 3 (to run the bootstrap script)
- At least 16 GB of RAM
- A GPU able to run both qwen2.5:14b and qwen2.5:7b

Ollama is installed for you by the bootstrap script if it is missing. The two chat
models and the embedding model are pulled on first run, which takes a while and a few
gigabytes of disk.

## Installation

```bash
git clone https://github.com/doriandamian/AI-Drug-Discovery
cd AI-Drug-Discovery
cp .env.example .env      # then edit .env and set NEO4J_PASSWORD
python start.py
```

`start.py` is the intended entry point. It checks that Docker is running, installs and
starts Ollama if needed, pulls `qwen2.5:14b`, `qwen2.5:7b`, and `nomic-embed-text`,
then brings up the Compose stack. Set `NEO4J_PASSWORD` in `.env` before you start; the
backend refuses to boot without it.

Once it reports the system is up:

| Service | URL |
|---|---|
| Web UI (Angular) | http://localhost:4200 |
| API (FastAPI) | http://localhost:8000 |
| Neo4j browser | http://localhost:7474 |

Press Ctrl+C in the terminal running `start.py` to stop the stack and release the
containers.

## Configuration

Configuration is read from `.env` (see `.env.example`). Everything except the Neo4j
password has a working default.

| Variable | Default | Purpose |
|---|---|---|
| `NEO4J_PASSWORD` | none (required) | Neo4j password; the backend will not start without it |
| `NEO4J_USERNAME` | `neo4j` | Neo4j user |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection URI |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Where the backend reaches Ollama |
| `MANAGER_MODEL` | `qwen2.5:14b` | Supervisor model |
| `SUBAGENT_MODEL` | `qwen2.5:7b` | Specialist model; set to `qwen2.5:14b` to run a single model |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | Reranker for hybrid retrieval |
| `NCBI_EMAIL` | `research@example.com` | Contact address sent to NCBI E-utilities |
| `SEMANTIC_SCHOLAR_API_KEY` | empty | Optional key to raise Semantic Scholar rate limits |
| `ALLOWED_ORIGINS` | `http://localhost:4200` | CORS origins allowed to call the API |

## Testing

The backend has a pytest suite covering the chemistry tools, feature extraction,
graph normalization, orchestrator routing, the SMILES guard, molecular design, and the
two-tier evaluation judge.

```bash
cd backend
pip install -r requirements.txt
pytest
```

## Evaluation

The agent is evaluated end to end against a 300-question pharmacology benchmark that
covers molecular properties, toxicity, literature retrieval, knowledge-graph
reasoning, generative design, and a set of hallucination probes that check whether
the agent refuses to invent compounds, citations, or measurements.

Correctness is scored in two tiers: a deterministic rubric (exact facts, required
caveats, correct tool routing) acts as a strict regression guard, and an LLM judge
re-scores only the deterministic failures, rescuing answers that are correct but
phrased differently than the rubric anticipated.

Most recent run (`qwen2.5:14b`, 300 questions, 2 July 2026):

| Metric | Result |
|---|---|
| Automated pass rate | 275 / 283 (97.2%) |
| Hallucination-probe pass rate | 54 / 55 (98.2%) |
| Mean latency (p50 / p95) | 129.6s (110.6s / 258.8s) |
| Mean LLM hops per question | 2.03 |
| Ungrounded-SMILES guard interventions | 0 |

Pass rate by category (review items are manual-confirmation probes that also run the
automated checks):

| Category | Pass | Fail | Manual review | Total |
|---|---|---|---|---|
| Properties | 79 | 0 | 0 | 79 |
| Toxicity | 53 | 0 | 0 | 53 |
| Literature | 46 | 2 | 0 | 48 |
| Graph | 35 | 0 | 0 | 35 |
| Design | 30 | 0 | 0 | 30 |
| Hallucination | 38 | 1 | 16 | 55 |

Of 39 answers the deterministic rubric marked as failures, the LLM judge confirmed 31
as actually correct (different wording, numeric precision, or ordering) and upheld 8.

## Use Cases

- Drug candidate screening and property analysis
- Toxicity and drug-likeness assessment
- Literature-grounded mechanism and evidence lookup
- Target and indication exploration over a knowledge graph
- Generative design of analogs from a known lead compound

## Data Sources

The toxicity model is trained on the Tox21 and ClinTox datasets. At query time the
agent draws on ChEMBL for bioactivity and target data, PubChem for compound records,
and PubMed and Semantic Scholar for literature. Please respect each source's terms of
use; NCBI and Semantic Scholar in particular expect a contact email and rate limiting,
both of which are configurable in `.env`.

## Model Integrity

The trained toxicity model ships with a SHA-256 sidecar (`toxicity_model.pkl.sha256`).
The hash is checked before the model is loaded, so a corrupted or swapped file fails
loudly instead of producing silent, wrong predictions. Retraining regenerates the
sidecar.

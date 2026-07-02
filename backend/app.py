import os
import json
import time
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from core.logging_config import setup_logging
import core.database as db
import core.llm_config as llm
import rag.ingest as ingest
from rag.preload_corpus import run_if_needed as preload_corpus
from rag.retriever import warmup as warmup_retriever
import agents.orchestrator as _orch
from agents.specialists import build_specialists
from agents import smiles_guard
from pydantic import BaseModel
import ml.train_multitask as ml_trainer
from tools.toxicity_predictor import reload_bundle as reload_tox_bundle

setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "toxicity_model.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Backend starting...")

    logger.info("Waiting for Neo4j and Ollama concurrently...")
    await asyncio.gather(
        asyncio.to_thread(db.wait_for_neo4j),
        asyncio.to_thread(llm.wait_for_ollama),
    )
    logger.info("Connected to Neo4j and Ollama")

    logger.info("Compiling specialist agents...")
    build_specialists()
    logger.info("Specialist agents ready.")

    logger.info("Compiling supervisor graph...")
    _orch.build_orchestrator()
    logger.info("Supervisor graph ready.")

    logger.info("Setting up local database...")
    ingest.build_vector_store()

    logger.info("Checking corpus preload...")
    preload_corpus()

    logger.info("Ensuring RAG indexes...")
    ingest.ensure_indexes()

    logger.info("Warming up RAG retriever (vector store + reranker)...")
    await asyncio.to_thread(warmup_retriever)
    logger.info("RAG retriever ready.")

    logger.info("Checking toxicity model...")
    if not os.path.exists(MODEL_PATH):
        logger.info("Toxicity model not found. Training new model...")
        ml_trainer.train()
        reload_tox_bundle()
        logger.info("Toxicity model trained and loaded.")
    else:
        logger.info("Using existing toxicity model.")

    logger.info("Warming up agent models (supervisor + sub-agents into Ollama)...")
    _orch.warmup()
    logger.info("Agent models ready.")

    logger.info("Startup complete")

    yield

    logger.info("Backend shutting down...")

app = FastAPI(title="AI Drug Discovery Backend", version="1.0", lifespan=lifespan)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:4200")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

class ChatRequest(BaseModel):
    message: str

_chat_lock = asyncio.Semaphore(1)


@app.get("/")
async def root():
    return {"status": "Backend is running!"}

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not _chat_lock.locked():
        await _chat_lock.acquire()
    else:
        raise HTTPException(status_code=429, detail="A request is already in progress. Please wait.")

    if not request.message or not request.message.strip():
        _chat_lock.release()
        raise HTTPException(status_code=422, detail="Message must not be empty.")
    if len(request.message) > 8000:
        _chat_lock.release()
        raise HTTPException(status_code=422, detail="Message exceeds maximum length of 8000 characters.")
    if _orch.orchestrator is None:
        _chat_lock.release()
        raise HTTPException(status_code=503, detail="Agent pipeline is not ready. Try again in a moment.")

    try:
        smiles_guard.reset()
        smiles_guard.record_user_message(request.message)
    except Exception as e:
        logger.exception("SMILES guard initialisation failed")
        _chat_lock.release()
        raise HTTPException(status_code=500, detail=str(e)) from e

    logger.info("Received message: %s", request.message)
    start_time = time.time()
    inputs = {"messages": [("user", request.message)]}

    async def event_stream():
        final_content = ""
        streaming_msg_id = None
        try:
            async for mode, data in _orch.orchestrator.astream(
                inputs,
                stream_mode=["updates", "messages"],
                config={"recursion_limit": 15},
            ):
                if mode == "updates":
                    for node, payload in data.items():
                        for msg in payload.get("messages", []):
                            if getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    logger.info("Tool call: %s args=%s", tc["name"], tc.get("args"))
                                names = [tc["name"] for tc in msg.tool_calls]
                                yield _sse({"type": "tool_call", "tools": names})
                            elif msg.type == "tool":
                                logger.info("Tool result [%s]:\n%s", msg.name, msg.content)
                                yield _sse({"type": "tool_result", "name": msg.name})
                            elif msg.type == "ai" and not msg.tool_calls and msg.content:
                                final_content = msg.content

                elif mode == "messages":
                    token, meta = data
                    if (
                        meta.get("langgraph_node") == "supervisor"
                        and getattr(token, "content", "")
                        and not getattr(token, "tool_call_chunks", None)
                    ):
                        msg_id = getattr(token, "id", None)
                        if streaming_msg_id is not None and msg_id != streaming_msg_id:
                            yield _sse({"type": "reset"})
                        streaming_msg_id = msg_id
                        yield _sse({"type": "token", "content": token.content})

            final_content, removed = smiles_guard.sanitize(final_content)
            if removed:
                logger.warning("Stripped %d ungrounded SMILES from answer: %s", len(removed), removed)

            elapsed = time.time() - start_time
            logger.info("Final response:\n%s", final_content)
            logger.info("Total processing time: %.2fs", elapsed)
            yield _sse({"type": "final", "message": final_content, "time": f"{elapsed:.2f}"})
            yield _sse({"type": "done"})

        except Exception as e:
            logger.exception("Agent error")
            yield _sse({"type": "error", "detail": str(e), "code": type(e).__name__})
        finally:
            _chat_lock.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
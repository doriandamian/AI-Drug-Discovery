import os
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import core.database as db
import core.llm_config as llm
import rag.ingest as ingest
from agents.orchestrator import orchestrator
from pydantic import BaseModel
import ml.train_model as ml_trainer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "toxicity_model.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("BACKEND STARTING...")

    print("WAITING FOR NEO4J...")
    db.wait_for_neo4j()
    print("CONNECTED TO NEO4J")

    print("WAITING FOR OLLAMA...")
    llm.wait_for_ollama()
    print("CONNECTED TO OLLAMA")

    print("SETTING UP LOCAL DATABASE...")
    ingest.build_vector_store()

    print("CHECKING TOXICITY MODEL...")
    if not os.path.exists(MODEL_PATH):
        print("TOXICITY MODEL NOT FOUND. TRAINING NEW MODEL...")
        ml_trainer.train()
    else:
        print("USING EXISTING TOXICITY MODEL.")

    print("STARTUP IS COMPLETE")

    yield

    print("BACKEND SHUTTING DOWN...")

app = FastAPI(title="AI Drug Discovery Backend", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"status": "Backend is running!"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"Received message: {request.message}")
    start_time = time.time()
    inputs = {"messages": [("user", request.message)]}

    final_content = ""

    events = orchestrator.stream(inputs, stream_mode="values", config={"recursion_limit": 15})

    SEP = "─" * 52

    step = 0
    step_start = time.time()

    try:
        for event in events:
            if "messages" not in event:
                continue

            step += 1
            msg = event["messages"][-1]
            elapsed = time.time() - step_start
            step_start = time.time()

            print(f"\n{SEP}")

            if msg.type == "human":
                print(f"STEP {step} │ USER")
                print(SEP)
                print(msg.content[:300])

            elif msg.type == "ai" and msg.tool_calls:
                n = len(msg.tool_calls)
                print(f"STEP {step} │ MANAGER → calling {n} tool{'s' if n > 1 else ''}")
                print(SEP)
                for tc in msg.tool_calls:
                    args = ", ".join(f'"{v}"' for v in tc["args"].values()) if tc["args"] else ""
                    print(f"  ▸ {tc['name']}({args})")

            elif msg.type == "tool":
                print(f"STEP {step} │ TOOL RESULT ← {msg.name}  ({elapsed:.2f}s)")
                print(SEP)
                print(msg.content[:600])

            elif msg.type == "ai" and not msg.tool_calls:
                print(f"STEP {step} │ FINAL RESPONSE  ({elapsed:.2f}s)")
                print(SEP)
                print(msg.content[:400])
                if msg.content:
                    final_content = msg.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

    return {
        "status": "success",
        "message": final_content,
        "time": f"{elapsed_time:.2f}",
    }
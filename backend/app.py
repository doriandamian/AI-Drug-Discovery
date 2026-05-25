import os
import time
from fastapi import FastAPI
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
    allow_credentials=True,
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

    events = orchestrator.stream(inputs, stream_mode="values")

    print("Processing response...")

    for event in events:
        if "messages" in event:
            last_message = event["messages"][-1]
            print(f"\n[STEP: {last_message.type.upper()}]")
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print(f"Tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
            elif last_message.type == "tool":
                print(f"📥 DATE PRIMITE DE LA: {last_message.name}")
                print(last_message.content) # Afișăm TOT conținutul găsit (lucrări, date)
            else:
                print(f"Content: {last_message.content[:200]}...")
            
            if last_message.type == "ai" and not last_message.tool_calls:
                final_content = last_message.content

    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

    return {
        "status": "success",
        "message": final_content,
        "time": f"{elapsed_time:.2f}",
    }
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import core.database as db
import core.llm_config as llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("BACKEND STARTING...")

    print("WAITING FOR NEO4J...")
    db.wait_for_neo4j()
    print("CONNECTED TO NEO4J")

    print("WAITING FOR OLLAMA...")
    llm.wait_for_ollama()
    print("CONNECTED TO OLLAMA")

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

@app.get("/")
async def root():
    return {"status": "Backend is running!"}

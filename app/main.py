"""FastAPI app: single /research endpoint the CLI client talks to."""
from fastapi import FastAPI
from pydantic import BaseModel

from app.config import config
from app.local_docs import store
from app.orchestrator import run_research

app = FastAPI(title="Deep Research Agent")


class ResearchRequest(BaseModel):
    question: str
    clarification_answers: dict[str, str] | None = None


@app.on_event("startup")
def startup():
    store.load_cache()
    try:
        stats = store.ingest()
        print(f"[startup] Local docs ready: {stats}")
    except FileNotFoundError as e:
        print(f"[startup] Warning: {e}")


@app.post("/research")
async def research(req: ResearchRequest):
    return await run_research(req.question, req.clarification_answers)


@app.post("/reindex")
def reindex():
    """Force a re-scan/re-embed of the docs directory (also picks up new files)."""
    return store.ingest(force=False)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(store.chunks)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

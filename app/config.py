"""Central configuration, loaded from environment variables / .env file."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Anthropic
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    PLANNER_MODEL: str = os.environ.get("PLANNER_MODEL", "claude-sonnet-4-6")
    SUBAGENT_MODEL: str = os.environ.get("SUBAGENT_MODEL", "claude-sonnet-4-6")
    SYNTHESIZER_MODEL: str = os.environ.get("SYNTHESIZER_MODEL", "claude-opus-4-1")
    CLARIFIER_MODEL: str = os.environ.get("CLARIFIER_MODEL", "claude-sonnet-4-6")

    # Local docs
    DOCS_DIR: str = os.environ.get("DOCS_DIR", "./docs")
    DOCS_CACHE_PATH: str = os.environ.get("DOCS_CACHE_PATH", "./.docs_cache.pkl")
    CHUNK_SIZE_TOKENS: int = int(os.environ.get("CHUNK_SIZE_TOKENS", "650"))
    CHUNK_OVERLAP_TOKENS: int = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "80"))
    TOP_K: int = int(os.environ.get("TOP_K", "5"))

    # Ollama embeddings
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "nomic-embed-text")

    # Sub-agent loop controls
    MAX_SUBAGENT_ITERATIONS: int = int(os.environ.get("MAX_SUBAGENT_ITERATIONS", "4"))
    SUBAGENT_TIMEOUT_SECONDS: int = int(os.environ.get("SUBAGENT_TIMEOUT_SECONDS", "60"))

    # API
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))


config = Config()

"""
Local document ingestion + retrieval.

Parses PDF / MD / TXT / JSON files from a directory, chunks them, embeds
each chunk via a local Ollama model, and holds everything in memory for
cosine-similarity search. Suitable for "dozens of docs" scale — no vector
DB required.

Cache is persisted to disk so we don't re-embed unchanged files on restart.
"""
from __future__ import annotations

import json
import pickle
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

from app.config import config


@dataclass
class Chunk:
    text: str
    source: str           # filename
    chunk_index: int
    vector: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def _read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))

    def flatten(obj, prefix=""):
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                lines.extend(flatten(v, f"{prefix}{k}."))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                lines.extend(flatten(v, f"{prefix}{i}."))
        else:
            lines.append(f"{prefix.rstrip('.')}: {obj}")
        return lines

    return "\n".join(flatten(data))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix in (".md", ".txt"):
        return _read_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Chunking (approximate token count via word count * 1.3; good enough here)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    # Approximate: 1 token ~= 0.75 words, so convert token targets to words
    words_per_chunk = max(int(chunk_size * 0.75), 50)
    words_overlap = max(int(overlap * 0.75), 0)

    chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - words_overlap
    return chunks


# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    # Ollama's embeddings endpoint is per-text; loop (fine at this scale).
    return [embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Store: in-memory + disk cache
# ---------------------------------------------------------------------------

class LocalDocStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._file_hashes: dict[str, str] = {}

    def _file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load_cache(self):
        cache_path = Path(config.DOCS_CACHE_PATH)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            self._file_hashes = data["file_hashes"]

    def save_cache(self):
        cache_path = Path(config.DOCS_CACHE_PATH)
        with open(cache_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "file_hashes": self._file_hashes}, f)

    def ingest(self, docs_dir: str | None = None, force: bool = False):
        """Scan docs_dir, (re)embed any new/changed files, update store."""
        docs_dir = Path(docs_dir or config.DOCS_DIR)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        supported = {".pdf", ".md", ".txt", ".json"}
        files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in supported]

        changed_files = []
        for path in files:
            file_hash = self._file_hash(path)
            key = str(path.resolve())
            if force or self._file_hashes.get(key) != file_hash:
                changed_files.append((path, file_hash))

        if not changed_files:
            return {"ingested": 0, "total_chunks": len(self.chunks)}

        # Drop stale chunks for changed files, then re-add
        changed_sources = {str(p.resolve()) for p, _ in changed_files}
        self.chunks = [c for c in self.chunks if c.source not in changed_sources]

        for path, file_hash in changed_files:
            key = str(path.resolve())
            try:
                text = parse_file(path)
            except Exception as e:
                print(f"[local_docs] Failed to parse {path}: {e}")
                continue

            pieces = chunk_text(text, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
            if not pieces:
                continue

            vectors = embed_batch(pieces)
            for i, (piece, vec) in enumerate(zip(pieces, vectors)):
                self.chunks.append(Chunk(text=piece, source=key, chunk_index=i, vector=vec))

            self._file_hashes[key] = file_hash

        self.save_cache()
        return {"ingested": len(changed_files), "total_chunks": len(self.chunks)}

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        if not self.chunks:
            return []
        top_k = top_k or config.TOP_K
        query_vec = np.array(embed(query))
        matrix = np.array([c.vector for c in self.chunks])

        # Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        scores = matrix_norm @ query_norm

        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append({
                "text": c.text,
                "source": Path(c.source).name,
                "chunk_index": c.chunk_index,
                "score": float(scores[idx]),
            })
        return results


# Module-level singleton store, initialized once by the app on startup.
store = LocalDocStore()

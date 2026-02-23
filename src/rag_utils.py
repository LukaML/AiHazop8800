"""RAG (Retrieval-Augmented Generation) support for the HAZOP pipeline.

Provides document loading, chunking, embedding-based indexing, and similarity
search.  Three embedding backends are supported:
  - local:  TF-IDF via scikit-learn (offline, no API calls)
  - openai: OpenAI text-embedding API
  - gemini: Google Gemini embedding API

The pipeline calls this module from graph_full.py RAG nodes to build context
that is injected into LLM prompts at each stage.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    ENGLISH_STOP_WORDS = set()
    cosine_similarity = None

# HAZOP-relevant terms that sklearn's English stopword list would remove
_HAZOP_KEEP = frozenset({
    "no", "not", "more", "less", "before", "after", "other", "part",
    "system", "down", "off", "out", "between", "through",
    "during", "above", "below", "under", "over", "further",
})


def _hazop_stopwords() -> list:
    """English stopwords minus HAZOP-relevant terms."""
    if TfidfVectorizer is None:
        return []
    return sorted(ENGLISH_STOP_WORDS - _HAZOP_KEEP)


# -----------------------------
# Types
# -----------------------------

@dataclass
class OpenAIEmbeddingIndex:
    """Marker object used as `vec` when we build an OpenAI-embedding index."""
    model: str


@dataclass
class GeminiEmbeddingIndex:
    """Marker object used as `vec` when we build a Gemini-embedding index."""
    model: str


# ---------------------------------------------------------------------------
# File loading — reads TXT/MD, CSV, XLSX/XLS, and PDF into raw text strings.
# Directories are scanned recursively for supported extensions.
# ---------------------------------------------------------------------------

_SUPPORTED_EXTS = {".txt", ".md", ".csv", ".xlsx", ".xls", ".pdf"}


def _iter_files(paths: Sequence[str]) -> Iterable[Path]:
    for p in paths:
        if not p:
            continue
        pp = Path(p).expanduser()
        if pp.is_dir():
            for f in sorted(pp.rglob("*")):
                if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS:
                    yield f
        elif pp.is_file():
            if pp.suffix.lower() in _SUPPORTED_EXTS:
                yield pp


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\t ]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_csv(path: Path) -> str:
    if pd is None:
        raise ImportError("pandas is required to read CSV files. Install it with: pip install pandas")
    df = pd.read_csv(path)
    return df.to_csv(index=False)


def _read_xlsx(path: Path) -> str:
    if pd is None:
        raise ImportError("pandas is required to read Excel files. Install it with: pip install pandas")
    xls = pd.ExcelFile(path)
    parts: List[str] = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        parts.append(f"=== SHEET: {sheet} ===\n" + df.to_csv(index=False))
    return "\n\n".join(parts)


def _read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("PyPDF2 is required to read PDF files. Install it with: pip install PyPDF2")
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)


def _read_any(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return _read_txt(path)
    if ext == ".csv":
        return _read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return _read_xlsx(path)
    if ext == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported extension: {ext}")


# ---------------------------------------------------------------------------
# Chunking — splits long text into overlapping chunks for embedding.  Splits
# on paragraph boundaries first to keep semantic coherence, then uses a
# character-level overlap to avoid losing context at chunk boundaries.
# ---------------------------------------------------------------------------


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into roughly `max_chars` chunks with `overlap`.

    Chunking by paragraphs first to keep coherence.
    """
    text = _clean_text(text)
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        joined = "\n\n".join(buf).strip()
        if joined:
            chunks.append(joined)
        # overlap: keep last N chars from the joined text as the start of next chunk
        if overlap > 0 and joined:
            tail = joined[-overlap:]
            buf = [tail]
            buf_len = len(tail)
        else:
            buf = []
            buf_len = 0

    for p in paras:
        if buf_len + len(p) + 2 <= max_chars:
            buf.append(p)
            buf_len += len(p) + 2
        else:
            flush()
            buf.append(p)
            buf_len = len(p)

    flush()
    return chunks


def load_documents(paths: Sequence[str], *, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Load a list of files (or directories) and return a list of text chunks."""
    docs: List[str] = []
    seen_hashes: set[str] = set()
    for f in _iter_files(paths):
        try:
            raw = _read_any(f)
        except Exception as e:
            logger.warning("Failed to load file %s: %s", f, e)
            continue
        raw = _clean_text(raw)
        if not raw:
            continue
        for ch in chunk_text(raw, max_chars=chunk_chars, overlap=overlap):
            h = hashlib.sha256(ch.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            # Light provenance marker helps the LLM ground answers
            docs.append(f"[SOURCE: {f.name}]\n{ch}")
    return docs


# ---------------------------------------------------------------------------
# Embedding backends — TF-IDF (local/offline) and API-based (OpenAI, Gemini).
# All backends produce a (vec, mat) pair: vec identifies the backend for
# search dispatch, mat is the document embedding matrix.
# ---------------------------------------------------------------------------


def _openai_client():
    # openai SDK is already used in your project (llm_client.py)
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
    return OpenAI(api_key=key)


def _openai_embed(texts: List[str], model: str):
    if np is None:
        raise ImportError("numpy is required for embeddings. Install it with: pip install numpy")
    client = _openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.asarray(vecs, dtype=np.float32)


_GEMINI_EMBED_BATCH = 100  # Gemini batchEmbedContents hard limit


def _gemini_embed(texts: List[str], model: str):
    if np is None:
        raise ImportError("numpy is required for embeddings. Install it with: pip install numpy")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")

    all_vecs: list = []
    for i in range(0, len(texts), _GEMINI_EMBED_BATCH):
        batch = texts[i : i + _GEMINI_EMBED_BATCH]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:batchEmbedContents"
            f"?key={key}"
        )
        body = json.dumps({
            "requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
                for t in batch
            ]
        }).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"Gemini embedding API error {e.code}: {err_body}"
            ) from e
        all_vecs.extend(emb["values"] for emb in data["embeddings"])
    return np.asarray(all_vecs, dtype=np.float32)


def build_index(
    docs: Sequence[str],
    *,
    embedder: str = "local",
    openai_model: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Build an index.

    Returns (vec, mat):
    - local: vec=TfidfVectorizer, mat=sparse matrix
    - openai: vec=OpenAIEmbeddingIndex(model=...), mat=dense np.ndarray
    - gemini: vec=GeminiEmbeddingIndex(model=...), mat=dense np.ndarray
    """
    embedder = (embedder or "local").strip().lower()
    if embedder not in {"local", "openai", "gemini"}:
        raise ValueError("embedder must be 'local', 'openai', or 'gemini'")

    if embedder == "local" and TfidfVectorizer is None:
        raise ImportError(
            "scikit-learn is required for local RAG. Install it with: pip install scikit-learn"
        )
    if embedder in {"openai", "gemini"} and np is None:
        raise ImportError(
            "numpy is required for embedding-based RAG. Install it with: pip install numpy"
        )

    if not docs:
        # Return empty but valid structures that won't fail on search
        if embedder == "local":
            # Fit on a dummy doc with real content to avoid empty vocabulary error
            v = TfidfVectorizer(stop_words=_hazop_stopwords())
            v.fit(["placeholder_token"])
            # Return an empty sparse matrix (0 rows)
            from scipy.sparse import csr_matrix
            empty_mat = csr_matrix((0, len(v.vocabulary_)))
            return v, empty_mat
        elif embedder == "openai":
            return OpenAIEmbeddingIndex(model=openai_model or "text-embedding-3-small"), np.zeros((0, 1), dtype=np.float32)
        else:  # gemini
            return GeminiEmbeddingIndex(model="gemini-embedding-001"), np.zeros((0, 1), dtype=np.float32)

    if embedder == "local":
        vec = TfidfVectorizer(stop_words=_hazop_stopwords(), max_features=50000)
        mat = vec.fit_transform(list(docs))
        return vec, mat

    if embedder == "gemini":
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
        mat = _gemini_embed(list(docs), model=model)
        # Normalize for cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = mat / norms
        return GeminiEmbeddingIndex(model=model), mat

    # OpenAI embeddings
    model = (openai_model or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    mat = _openai_embed(list(docs), model=model)
    # Normalize for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms
    return OpenAIEmbeddingIndex(model=model), mat


# ---------------------------------------------------------------------------
# Similarity search — dispatches to the correct backend based on the vec
# type (TfidfVectorizer, OpenAIEmbeddingIndex, or GeminiEmbeddingIndex).
# Returns the top_k most similar document chunks above min_similarity.
# ---------------------------------------------------------------------------

def search(
    query: str,
    vec: Any,
    mat: Any,
    docs: Sequence[str],
    *,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> List[str]:
    """Return top_k doc chunks for query."""
    q = (query or "").strip()
    if not q or not docs:
        return []
    top_k = max(1, int(top_k or 3))

    # Local TF-IDF
    if isinstance(vec, TfidfVectorizer):
        qv = vec.transform([q])
        sims = cosine_similarity(qv, mat).ravel()
        idx = np.argsort(-sims)[:top_k]
        logger.debug("RAG search (%s): query=%.60s top_sims=%s",
                     type(vec).__name__, q,
                     [round(float(sims[i]), 3) for i in idx[:top_k] if sims[i] >= min_similarity])
        filtered = [round(float(sims[i]), 3) for i in idx[:top_k] if sims[i] < min_similarity]
        if filtered:
            logger.debug("RAG filtered out %d chunks below min_sim=%.2f: %s",
                         len(filtered), min_similarity, filtered)
        return [docs[i] for i in idx if sims[i] >= min_similarity]

    # OpenAI embeddings
    if isinstance(vec, OpenAIEmbeddingIndex):
        qemb = _openai_embed([q], model=vec.model)
        qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-12)
        sims = (mat @ qemb[0]).ravel() if isinstance(mat, np.ndarray) else np.asarray([])
        idx = np.argsort(-sims)[:top_k]
        logger.debug("RAG search (%s): query=%.60s top_sims=%s",
                     type(vec).__name__, q,
                     [round(float(sims[i]), 3) for i in idx[:top_k] if sims[i] >= min_similarity])
        filtered = [round(float(sims[i]), 3) for i in idx[:top_k] if sims[i] < min_similarity]
        if filtered:
            logger.debug("RAG filtered out %d chunks below min_sim=%.2f: %s",
                         len(filtered), min_similarity, filtered)
        return [docs[i] for i in idx if sims[i] >= min_similarity]

    # Gemini embeddings
    if isinstance(vec, GeminiEmbeddingIndex):
        qemb = _gemini_embed([q], model=vec.model)
        qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-12)
        sims = (mat @ qemb[0]).ravel() if isinstance(mat, np.ndarray) else np.asarray([])
        idx = np.argsort(-sims)[:top_k]
        logger.debug("RAG search (%s): query=%.60s top_sims=%s",
                     type(vec).__name__, q,
                     [round(float(sims[i]), 3) for i in idx[:top_k] if sims[i] >= min_similarity])
        filtered = [round(float(sims[i]), 3) for i in idx[:top_k] if sims[i] < min_similarity]
        if filtered:
            logger.debug("RAG filtered out %d chunks below min_sim=%.2f: %s",
                         len(filtered), min_similarity, filtered)
        return [docs[i] for i in idx if sims[i] >= min_similarity]

    # Unknown index type
    return []

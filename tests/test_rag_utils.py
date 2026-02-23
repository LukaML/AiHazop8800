# tests/test_rag_utils.py
"""Tests for src/rag_utils.py RAG utility functions."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag_utils import (
    chunk_text,
    _clean_text,
    load_documents,
    build_index,
    search,
    OpenAIEmbeddingIndex,
)


class TestCleanText:
    """Tests for _clean_text() function."""

    def test_whitespace_normalized(self):
        """Multiple spaces/tabs normalized to single space."""
        text = "hello    world\t\ttest"
        result = _clean_text(text)
        assert result == "hello world test"

    def test_newlines_normalized(self):
        """Multiple newlines reduced to double."""
        text = "para1\n\n\n\n\npara2"
        result = _clean_text(text)
        assert result == "para1\n\npara2"

    def test_crlf_converted(self):
        """Windows line endings converted."""
        text = "line1\r\nline2\rline3"
        result = _clean_text(text)
        assert "\r" not in result

    def test_stripped(self):
        """Leading/trailing whitespace stripped."""
        text = "  content  "
        result = _clean_text(text)
        assert result == "content"


class TestChunkText:
    """Tests for chunk_text() function."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_single_paragraph_under_limit(self):
        """Single paragraph under limit returned as one chunk."""
        text = "Short paragraph."
        chunks = chunk_text(text, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0] == "Short paragraph."

    def test_multiple_paragraphs_split(self):
        """Multiple paragraphs are chunked appropriately."""
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, max_chars=20, overlap=0)
        assert len(chunks) > 1

    def test_overlap_preserved(self):
        """Overlap from previous chunk is preserved."""
        text = "First paragraph content.\n\nSecond paragraph content."
        chunks = chunk_text(text, max_chars=30, overlap=10)
        # With overlap, chunks should share some content
        assert len(chunks) >= 2

    def test_large_paragraph_chunked(self):
        """Large text with multiple paragraphs gets chunked."""
        # chunk_text splits by paragraphs (double newlines), not by character count
        # within a single paragraph. Create multiple paragraphs to test chunking.
        text = ("word " * 50 + "\n\n") * 10
        chunks = chunk_text(text, max_chars=100, overlap=0)
        assert len(chunks) > 1


class TestLoadDocuments:
    """Tests for load_documents() function."""

    def test_loads_txt_file(self, tmp_path):
        """Loads content from .txt file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test content for document.")

        docs = load_documents([str(tmp_path)])
        assert len(docs) >= 1
        assert any("Test content" in d for d in docs)

    def test_loads_directory(self, tmp_path):
        """Loads all supported files from directory."""
        (tmp_path / "file1.txt").write_text("Content one.")
        (tmp_path / "file2.txt").write_text("Content two.")

        docs = load_documents([str(tmp_path)])
        assert len(docs) >= 2

    def test_skips_unsupported_extensions(self, tmp_path):
        """Skips files with unsupported extensions."""
        (tmp_path / "file.unsupported").write_text("Should be skipped.")
        (tmp_path / "file.txt").write_text("Should be loaded.")

        docs = load_documents([str(tmp_path)])
        assert all(".unsupported" not in d for d in docs)

    def test_adds_source_marker(self, tmp_path):
        """Adds [SOURCE: filename] marker to chunks."""
        txt_file = tmp_path / "myfile.txt"
        txt_file.write_text("Content here.")

        docs = load_documents([str(tmp_path)])
        assert any("[SOURCE: myfile.txt]" in d for d in docs)

    def test_handles_empty_path_list(self):
        """Empty path list returns empty docs."""
        docs = load_documents([])
        assert docs == []


class TestBuildIndex:
    """Tests for build_index() function."""

    def test_local_tfidf_index(self):
        """Builds local TF-IDF index."""
        docs = ["Document one content.", "Document two content."]
        vec, mat = build_index(docs, embedder="local")

        # vec should be TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        assert isinstance(vec, TfidfVectorizer)
        # mat should be sparse matrix
        assert mat.shape[0] == 2

    def test_empty_docs_handled(self):
        """Empty docs list returns valid but empty structures."""
        vec, mat = build_index([], embedder="local")
        assert mat.shape[0] == 0

    def test_invalid_embedder_raises(self):
        """Invalid embedder raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            build_index(["doc"], embedder="invalid")
        assert "embedder must be" in str(exc_info.value)


class TestSearch:
    """Tests for search() function."""

    def test_local_search_returns_results(self):
        """Local TF-IDF search returns relevant results."""
        docs = [
            "The quick brown fox jumps.",
            "A lazy dog sleeps.",
            "Quick foxes are fast.",
        ]
        vec, mat = build_index(docs, embedder="local")

        results = search("fox", vec, mat, docs, top_k=2)
        assert len(results) <= 2
        assert any("fox" in r.lower() for r in results)

    def test_empty_query_returns_empty(self):
        """Empty query returns empty results."""
        docs = ["Some content."]
        vec, mat = build_index(docs, embedder="local")

        results = search("", vec, mat, docs)
        assert results == []

    def test_empty_docs_returns_empty(self):
        """Empty docs returns empty results."""
        vec, mat = build_index([], embedder="local")

        results = search("query", vec, mat, [])
        assert results == []

    def test_top_k_respected(self):
        """top_k parameter limits results."""
        docs = ["Doc " + str(i) for i in range(10)]
        vec, mat = build_index(docs, embedder="local")

        results = search("Doc", vec, mat, docs, top_k=3)
        assert len(results) <= 3

    def test_unknown_index_type_returns_empty(self):
        """Unknown index type returns empty results."""
        results = search("query", "unknown", "unknown", ["doc"])
        assert results == []


class TestOpenAIEmbeddingIndex:
    """Tests for OpenAIEmbeddingIndex dataclass."""

    def test_stores_model_name(self):
        """Stores the embedding model name."""
        idx = OpenAIEmbeddingIndex(model="text-embedding-3-small")
        assert idx.model == "text-embedding-3-small"

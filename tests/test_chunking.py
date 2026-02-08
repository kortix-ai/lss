"""Tests for smart chunking — format-aware text splitting.

TDD: Written BEFORE the implementation.
Tests that _smart_chunk dispatches to format-specific chunkers.
"""

import pytest

from lss_store import _smart_chunk, _span_chunk


# ── Markdown chunking ────────────────────────────────────────────────────────


class TestMarkdownChunking:
    def test_splits_on_headings(self):
        """Markdown text with ## headings should be split at heading boundaries."""
        text = (
            "# Introduction\n\n"
            "This is the intro paragraph with enough words to be meaningful.\n\n"
            "## Architecture\n\n"
            "The system uses microservices with Kubernetes orchestration.\n"
            "Each service has its own database and API gateway.\n\n"
            "## Database\n\n"
            "We use PostgreSQL for the primary datastore.\n"
            "Redis is used for caching and session management.\n"
        )
        chunks = _smart_chunk(text, ".md")
        assert len(chunks) >= 2  # At least 2 sections

        # Each chunk should be a (chunk_type, text) tuple
        for chunk_type, chunk_text in chunks:
            assert isinstance(chunk_text, str)
            assert len(chunk_text) > 0

        # The heading text should be included with its section content
        all_text = " ".join(t for _, t in chunks)
        assert "Architecture" in all_text
        assert "Database" in all_text
        assert "microservices" in all_text

    def test_heading_stays_with_content(self):
        """Each chunk should start with or contain its heading."""
        text = (
            "## Setup\n\n"
            "Run npm install to get started.\n\n"
            "## Usage\n\n"
            "Import the module and call search().\n"
        )
        chunks = _smart_chunk(text, ".md")
        # At least one chunk should contain "Setup" with "npm install"
        found_setup = False
        for _, chunk_text in chunks:
            if "Setup" in chunk_text and "npm install" in chunk_text:
                found_setup = True
        assert found_setup, "Heading 'Setup' should be in same chunk as its content"

    def test_no_headings_fallback(self):
        """Markdown without headings should fall back to word-window chunking."""
        text = " ".join(f"word{i}" for i in range(500))
        chunks = _smart_chunk(text, ".md")
        assert len(chunks) > 1  # Should still chunk (word-window fallback)

    def test_short_markdown_single_chunk(self):
        """Short markdown should be a single chunk."""
        text = "# Title\n\nA short document."
        chunks = _smart_chunk(text, ".md")
        assert len(chunks) == 1

    def test_rst_same_as_markdown(self):
        """RST files should use the same heading-based chunking."""
        text = (
            "Introduction\n============\n\n"
            "This is the intro.\n\n"
            "Architecture\n============\n\n"
            "The system design.\n"
        )
        chunks = _smart_chunk(text, ".rst")
        assert len(chunks) >= 1


# ── Python chunking ──────────────────────────────────────────────────────────


class TestPythonChunking:
    def test_splits_on_functions(self):
        """Python code should be split on function/class boundaries."""
        text = (
            'def authenticate(user, password):\n'
            '    """Authenticate a user with password."""\n'
            '    token = create_jwt(user)\n'
            '    return validate(token)\n'
            '\n\n'
            'def authorize(token, resource):\n'
            '    """Check if token has access to resource."""\n'
            '    claims = decode_jwt(token)\n'
            '    return check_permissions(claims, resource)\n'
            '\n\n'
            'class UserManager:\n'
            '    """Manage user lifecycle."""\n'
            '    def create_user(self, name):\n'
            '        return User(name=name)\n'
        )
        chunks = _smart_chunk(text, ".py")
        assert len(chunks) >= 2

        all_text = " ".join(t for _, t in chunks)
        assert "authenticate" in all_text
        assert "authorize" in all_text
        assert "UserManager" in all_text

    def test_docstring_stays_with_function(self):
        """Function docstrings should be in the same chunk as the function."""
        text = (
            'def search(query, path):\n'
            '    """Search for files matching query in path.\n'
            '    \n'
            '    Uses BM25 + embedding hybrid search with RRF fusion.\n'
            '    """\n'
            '    results = bm25_search(query)\n'
            '    return rerank(results)\n'
        )
        chunks = _smart_chunk(text, ".py")
        # The docstring should be in the same chunk as the function
        found = False
        for _, chunk_text in chunks:
            if "search" in chunk_text and "BM25" in chunk_text:
                found = True
        assert found, "Docstring should be in same chunk as its function"

    def test_short_python_single_chunk(self):
        """Short Python file should be a single chunk."""
        text = "x = 1\nprint(x)\n"
        chunks = _smart_chunk(text, ".py")
        assert len(chunks) == 1

    def test_module_level_code_included(self):
        """Module-level code (imports, constants) should be included."""
        text = (
            'import os\nimport sys\n\n'
            'MAX_SIZE = 1024\n\n'
            'def main():\n    pass\n'
        )
        chunks = _smart_chunk(text, ".py")
        all_text = " ".join(t for _, t in chunks)
        assert "import os" in all_text or "MAX_SIZE" in all_text


# ── Plain text / default chunking ────────────────────────────────────────────


class TestDefaultChunking:
    def test_plain_text_uses_word_window(self):
        """Plain text should use the standard word-window chunking."""
        text = " ".join(f"word{i}" for i in range(500))
        chunks = _smart_chunk(text, ".txt")
        assert len(chunks) > 1
        for chunk_type, chunk_text in chunks:
            words = chunk_text.split()
            assert len(words) <= 220

    def test_json_uses_word_window(self):
        """JSON files should use word-window chunking."""
        text = " ".join(f"value{i}" for i in range(300))
        chunks = _smart_chunk(text, ".json")
        assert len(chunks) > 1

    def test_unknown_extension_uses_word_window(self):
        """Unknown extensions should use word-window chunking."""
        text = " ".join(f"data{i}" for i in range(300))
        chunks = _smart_chunk(text, ".xyz")
        assert len(chunks) > 1

    def test_empty_text(self):
        """Empty text should return no chunks."""
        chunks = _smart_chunk("", ".md")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        """Whitespace-only text should return no chunks."""
        chunks = _smart_chunk("   \n\n  \t  ", ".txt")
        assert len(chunks) == 0


# ── Backward compatibility ───────────────────────────────────────────────────


class TestSpanChunkCompat:
    def test_span_chunk_still_works(self):
        """The original _span_chunk function should still work for direct calls."""
        text = " ".join(f"word{i}" for i in range(500))
        chunks = _span_chunk(text, words_per_span=220, stride=200)
        assert len(chunks) > 1
        for chunk_type, chunk_text in chunks:
            assert chunk_type == "simple"

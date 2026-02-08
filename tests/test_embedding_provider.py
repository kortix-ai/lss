"""Tests for embedding provider abstraction (WS1).

Tests the config-based provider detection, local fastembed embedding,
and CLI config commands.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import lss_config


# ── Provider detection ───────────────────────────────────────────────────────


class TestProviderDetection:
    """_detect_provider() should auto-detect based on env/config/imports."""

    def test_env_override_openai(self, monkeypatch):
        """LSS_PROVIDER=openai overrides everything."""
        monkeypatch.setenv("LSS_PROVIDER", "openai")
        assert lss_config._detect_provider() == "openai"

    def test_env_override_local(self, monkeypatch):
        """LSS_PROVIDER=local overrides everything."""
        monkeypatch.setenv("LSS_PROVIDER", "local")
        assert lss_config._detect_provider() == "local"

    def test_config_file_provider(self, isolated_lss_dir, monkeypatch):
        """Provider from config.json should be used when no env override."""
        monkeypatch.delenv("LSS_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        cfg = lss_config.load_config()
        cfg["embedding_provider"] = "local"
        lss_config.save_config(cfg)

        assert lss_config._detect_provider() == "local"

    def test_auto_openai_when_key_set(self, monkeypatch):
        """With OPENAI_API_KEY set and no explicit override, prefer openai."""
        monkeypatch.delenv("LSS_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
        assert lss_config._detect_provider() == "openai"

    def test_auto_local_when_no_key_and_fastembed(self, monkeypatch):
        """Without API key but with fastembed installed, use local."""
        monkeypatch.delenv("LSS_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Config file doesn't exist in isolated_lss_dir, so no config override
        # fastembed is installed in our .venv, so it should be importable
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")

        result = lss_config._detect_provider()
        assert result == "local"


# ── Provider model/dim ───────────────────────────────────────────────────────


class TestProviderModelDim:
    """_provider_model_dim() returns correct values per provider."""

    def test_openai_model_dim(self, monkeypatch):
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "openai")
        model, dim = lss_config._provider_model_dim()
        assert model == lss_config.OPENAI_MODEL
        assert dim == lss_config.OPENAI_DIM
        assert dim == 256

    def test_local_model_dim(self, monkeypatch):
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "local")
        model, dim = lss_config._provider_model_dim()
        assert model == lss_config.LOCAL_MODEL
        assert dim == lss_config.LOCAL_DIM
        assert dim == 384


# ── VERSION_KEY incorporates provider ────────────────────────────────────────


class TestVersionKey:
    """VERSION_KEY must differ between providers to trigger re-embedding."""

    def test_version_key_contains_model(self):
        """VERSION_KEY should contain the model name."""
        model, _ = lss_config._provider_model_dim()
        assert model in lss_config.VERSION_KEY

    def test_version_key_contains_dim(self):
        """VERSION_KEY should contain the dimension."""
        _, dim = lss_config._provider_model_dim()
        assert str(dim) in lss_config.VERSION_KEY


# ── Local embedding function ─────────────────────────────────────────────────


class TestLocalEmbed:
    """_local_embed() should produce valid normalized vectors."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_fastembed(self):
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")

    def test_local_embed_basic(self):
        from semantic_search import _local_embed
        result = _local_embed(["hello world"])
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, lss_config.LOCAL_DIM)

    def test_local_embed_batch(self):
        from semantic_search import _local_embed
        texts = ["hello world", "search engine", "python programming"]
        result = _local_embed(texts)
        assert result is not None
        assert result.shape == (3, lss_config.LOCAL_DIM)

    def test_local_embed_normalized(self):
        from semantic_search import _local_embed
        result = _local_embed(["test normalization"])
        assert result is not None
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01, f"Vector should be unit-normalized, got norm={norm}"

    def test_local_embed_empty_returns_none(self):
        from semantic_search import _local_embed
        result = _local_embed([])
        assert result is None

    def test_local_embed_different_texts_different_vectors(self):
        from semantic_search import _local_embed
        result = _local_embed(["cat", "quantum physics"])
        assert result is not None
        sim = float(result[0] @ result[1])
        assert sim < 0.95, f"Very different texts should have low similarity, got {sim}"


# ── _embed dispatch ──────────────────────────────────────────────────────────


class TestEmbedDispatch:
    """_embed() should dispatch to the correct provider."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_fastembed(self):
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")

    def test_embed_dispatches_to_local(self, monkeypatch):
        import semantic_search
        monkeypatch.setattr(semantic_search, "EMBED_PROVIDER", "local")
        result = semantic_search._embed(["hello"])
        assert result is not None
        assert result.shape[1] == lss_config.LOCAL_DIM


# ── CLI config commands ──────────────────────────────────────────────────────


class TestConfigCLI:
    """Tests for 'lss config show' and 'lss config provider'."""

    @pytest.fixture(autouse=True)
    def _no_colors(self):
        from lss_cli import _C
        _C.set_enabled(False)
        yield
        _C.set_enabled(None)

    def test_config_show(self, capsys):
        from lss_cli import main
        rc = main(["config", "show"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "provider" in out.lower()

    def test_config_provider_local(self, capsys):
        from lss_cli import main
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")

        rc = main(["config", "provider", "local"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "local" in out.lower()

        # Verify it persisted
        cfg = lss_config.load_config()
        assert cfg.get("embedding_provider") == "local"

    def test_config_provider_openai(self, capsys, monkeypatch):
        from lss_cli import main
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        rc = main(["config", "provider", "openai"])
        assert rc == 0

    def test_config_provider_invalid(self, capsys):
        from lss_cli import main
        rc = main(["config", "provider", "bogus"])
        assert rc == 2

    def test_config_provider_local_without_fastembed(self, capsys, monkeypatch):
        """Setting local provider without fastembed should fail gracefully."""
        from lss_cli import main
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        rc = main(["config", "provider", "local"])
        assert rc == 1


# ── Check embedding provider ────────────────────────────────────────────────


class TestCheckEmbeddingProvider:
    """_check_embedding_provider() validation."""

    @pytest.fixture(autouse=True)
    def _no_colors(self):
        from lss_cli import _C
        _C.set_enabled(False)
        yield
        _C.set_enabled(None)

    def test_check_passes_with_local_and_fastembed(self, monkeypatch):
        from lss_cli import _check_embedding_provider
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "local")
        assert _check_embedding_provider() is True

    def test_check_passes_with_openai_and_key(self, monkeypatch):
        from lss_cli import _check_embedding_provider
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        assert _check_embedding_provider() is True

    def test_check_fails_openai_no_key(self, monkeypatch):
        from lss_cli import _check_embedding_provider
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _check_embedding_provider() is False

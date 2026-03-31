from __future__ import annotations


class TestIsModelAllowed:
    def test_allowlisted_model_is_permitted(self):
        from cindex.services.config import is_model_allowed

        assert is_model_allowed("sentence-transformers/all-MiniLM-L6-v2") is True

    def test_unknown_model_is_rejected_by_default(self):
        from cindex.services.config import is_model_allowed

        assert is_model_allowed("some-org/custom-model") is False

    def test_allow_any_flag_bypasses_allowlist(self):
        from cindex.services.config import is_model_allowed

        assert is_model_allowed("some-org/custom-model", allow_any=True) is True

    def test_env_var_bypasses_allowlist(self, monkeypatch):
        from cindex.services.config import is_model_allowed

        monkeypatch.setenv("CINDEX_ALLOW_ANY_MODEL", "1")
        assert is_model_allowed("some-org/custom-model") is True

    def test_env_var_true_string_bypasses_allowlist(self, monkeypatch):
        from cindex.services.config import is_model_allowed

        monkeypatch.setenv("CINDEX_ALLOW_ANY_MODEL", "true")
        assert is_model_allowed("some-org/custom-model") is True

    def test_env_var_unset_does_not_bypass(self, monkeypatch):
        from cindex.services.config import is_model_allowed

        monkeypatch.delenv("CINDEX_ALLOW_ANY_MODEL", raising=False)
        assert is_model_allowed("some-org/custom-model") is False

    def test_env_var_arbitrary_value_does_not_bypass(self, monkeypatch):
        from cindex.services.config import is_model_allowed

        monkeypatch.setenv("CINDEX_ALLOW_ANY_MODEL", "maybe")
        assert is_model_allowed("some-org/custom-model") is False

    def test_all_allowlisted_models_are_permitted(self):
        from cindex.services.config import ALLOWED_MODELS
        from cindex.services.config import is_model_allowed

        for model in ALLOWED_MODELS:
            assert is_model_allowed(model) is True, f"{model} should be allowed"


class TestServeCommandModelValidation:
    def test_serve_rejects_unlisted_model(self, monkeypatch):
        from cindex.cli.commands import serve

        monkeypatch.delenv("CINDEX_ALLOW_ANY_MODEL", raising=False)

        rc = serve.run(
            type(
                "Args",
                (),
                {
                    "directory": ".",
                    "host": "127.0.0.1",
                    "port": 9000,
                    "model": "unknown-org/mystery-model",
                    "cache_dir": None,
                    "sqlite_db": "data.db",
                    "no_preload": False,
                    "allow_any_model": False,
                    "cors_origins": None,
                    "rate_limit": 0,
                },
            )()
        )

        assert rc == 1

    def test_serve_accepts_unlisted_model_with_flag(self, monkeypatch, tmp_path):
        from cindex.cli.commands import serve

        monkeypatch.delenv("CINDEX_ALLOW_ANY_MODEL", raising=False)
        monkeypatch.setattr("cindex.server.app.create_app", lambda **kw: object())
        monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)

        rc = serve.run(
            type(
                "Args",
                (),
                {
                    "directory": ".",
                    "host": "127.0.0.1",
                    "port": 9000,
                    "model": "unknown-org/mystery-model",
                    "cache_dir": None,
                    "sqlite_db": str(tmp_path / "data.db"),
                    "no_preload": False,
                    "allow_any_model": True,
                    "cors_origins": None,
                    "rate_limit": 0,
                },
            )()
        )

        assert rc == 0

    def test_serve_accepts_unlisted_model_via_env_var(self, monkeypatch, tmp_path):
        from cindex.cli.commands import serve

        monkeypatch.setenv("CINDEX_ALLOW_ANY_MODEL", "1")
        monkeypatch.setattr("cindex.server.app.create_app", lambda **kw: object())
        monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)

        rc = serve.run(
            type(
                "Args",
                (),
                {
                    "directory": ".",
                    "host": "127.0.0.1",
                    "port": 9000,
                    "model": "unknown-org/mystery-model",
                    "cache_dir": None,
                    "sqlite_db": str(tmp_path / "data.db"),
                    "no_preload": False,
                    "allow_any_model": False,
                    "cors_origins": None,
                    "rate_limit": 0,
                },
            )()
        )

        assert rc == 0

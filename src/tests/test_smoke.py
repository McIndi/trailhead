from cindex import cli


def test_main_smoke(monkeypatch, capsys, caplog) -> None:
    recorded: dict[str, str | None] = {}

    def fake_generate_embedding(
        text: str, model_name: str, cache_folder: str | None = None
    ) -> list[float]:
        recorded["text"] = text
        recorded["model_name"] = model_name
        recorded["cache_folder"] = cache_folder
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        "cindex.cli.commands.embed.generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.delenv("CINDEX_CACHE_DIR", raising=False)

    assert (
        cli.main(
            [
                "embed",
                "hello world",
                "--model",
                "sentence-transformers/all-mpnet-base-v2",
            ]
        )
        == 0
    )
    assert recorded == {
        "text": "hello world",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "cache_folder": None,
    }
    assert capsys.readouterr().out.strip() == "[0.1, 0.2, 0.3]"
    assert "Using cache directory" in caplog.text

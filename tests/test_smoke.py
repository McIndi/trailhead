from cindex.cli import main


def test_main_smoke() -> None:
    assert main() == 0
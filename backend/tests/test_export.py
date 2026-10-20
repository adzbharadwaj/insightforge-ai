import sys

from insightforge.export import render


def test_homebrew_lib_dirs_are_added_once_and_existing_paths_kept(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(render.Path, "is_dir", lambda self: str(self) == "/opt/homebrew/lib")
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/custom/lib")
    render._add_homebrew_libs()
    render._add_homebrew_libs()
    assert render.os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == "/custom/lib:/opt/homebrew/lib"


def test_library_path_is_left_alone_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    render._add_homebrew_libs()
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in render.os.environ

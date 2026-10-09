"""Tests for binary discovery, path inspection, and verification."""

import platform
from pathlib import Path
import pytest

from quantforge.core.binary_manager import BinaryInfo, BinaryManager
from quantforge.core.config import settings


def test_binary_info_dataclass():
    info = BinaryInfo(name="quantize", found=True, path=Path("C:/tools/quantize.exe"), version="b3800")
    assert info.found is True
    assert info.name == "quantize"
    assert info.path == Path("C:/tools/quantize.exe")
    assert info.version == "b3800"


def test_executable_names_generation():
    mgr = BinaryManager()
    names = mgr._executable_names("quantize")
    is_win = platform.system().lower() == "windows"
    if is_win:
        assert "llama-quantize.exe" in names or "quantize.exe" in names
    else:
        assert "llama-quantize" in names or "quantize" in names


def test_search_paths_contain_custom_and_system(tmp_path: Path):
    custom_dir = tmp_path / "custom_bin"
    custom_dir.mkdir()
    mgr = BinaryManager(custom_dir=custom_dir)
    paths = mgr.get_search_paths()
    assert custom_dir in paths


def test_require_binary_raises_if_missing(tmp_path: Path):
    empty_dir = tmp_path / "empty_bin"
    empty_dir.mkdir()
    mgr = BinaryManager(custom_dir=empty_dir)
    # Monkey-patch get_search_paths to isolate
    mgr.get_search_paths = lambda: [empty_dir]
    with pytest.raises(FileNotFoundError, match="Required native tool 'nonexistent_tool' was not found"):
        mgr.require_binary("nonexistent_tool")


def test_binary_discovery_with_installed_binaries():
    mgr = BinaryManager()
    status = mgr.get_status()
    assert "quantize" in status
    assert "imatrix" in status
    assert "perplexity" in status
    assert "bench" in status
    # We downloaded official binaries into settings.binaries_path
    if status["quantize"].found:
        assert status["quantize"].path.is_file()

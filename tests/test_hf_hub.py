"""Tests for Hugging Face Hub integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from quantforge.formats.hf_hub import HFHubManager


def test_hf_hub_init():
    mgr = HFHubManager(token="test_token_123")
    assert mgr.token == "test_token_123"
    assert mgr.api is not None


@patch("quantforge.formats.hf_hub.HfApi.list_repo_files")
def test_list_gguf_files(mock_list):
    mock_list.return_value = ["config.json", "model-f16.gguf", "tokenizer.json", "model-q4km.GGUF"]
    mgr = HFHubManager()
    ggufs = mgr.list_gguf_files("test/repo")

    assert len(ggufs) == 2
    assert "model-f16.gguf" in ggufs
    assert "model-q4km.GGUF" in ggufs


@patch("quantforge.formats.hf_hub.HfApi.list_repo_files")
@patch("quantforge.formats.hf_hub.hf_hub_download")
def test_pull_model_auto_select(mock_download, mock_list, tmp_path: Path):
    mock_list.return_value = ["model-q4.gguf", "model-fp16.gguf"]
    out_file = tmp_path / "model-fp16.gguf"
    out_file.write_bytes(b"GGUF_TEST")
    mock_download.return_value = str(out_file)

    mgr = HFHubManager()
    res = mgr.pull_model("test/repo", target_dir=tmp_path)

    assert res == out_file
    mock_download.assert_called_once()
    assert mock_download.call_args[1]["filename"] == "model-fp16.gguf"


@patch("quantforge.formats.hf_hub.HfApi.create_repo")
@patch("quantforge.formats.hf_hub.HfApi.upload_folder")
def test_push_models_directory(mock_upload_folder, mock_create_repo, tmp_path: Path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "model-q4.gguf").write_bytes(b"GGUF")
    (dist_dir / "README.md").write_text("# Model Card")

    mgr = HFHubManager()
    url = mgr.push_models("test-org/model-gguf", source_path=dist_dir)

    assert url == "https://huggingface.co/test-org/model-gguf"
    mock_create_repo.assert_called_once()
    mock_upload_folder.assert_called_once()

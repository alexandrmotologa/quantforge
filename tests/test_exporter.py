"""Tests for Ollama and InferOps exporter."""

from pathlib import Path
import pytest
import yaml

from quantforge.formats.exporter import ModelExporter
from tests.helpers import write_synthetic_gguf


def test_model_exporter_generates_modelfile_and_inferops(tmp_path: Path):
    model_path = tmp_path / "mistral-7b-q4km.gguf"
    write_synthetic_gguf(model_path)

    exporter = ModelExporter(model_path)
    pkg = exporter.export(output_dir=tmp_path / "exports", system_prompt="You are a helpful coding assistant.")

    assert pkg.model_name == "mistral-7b-q4km"
    assert "FROM " in pkg.modelfile_content
    assert "PARAMETER num_ctx 4096" in pkg.modelfile_content
    assert "You are a helpful coding assistant." in pkg.modelfile_content

    # Check files written
    modelfile = tmp_path / "exports" / "Modelfile"
    assert modelfile.is_file()
    assert 'PARAMETER stop "<|eot_id|>"' in modelfile.read_text(encoding="utf-8")

    yaml_file = tmp_path / "exports" / "inferops.yaml"
    assert yaml_file.is_file()
    loaded_yaml = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    assert loaded_yaml["service"]["engine"] == "llama.cpp"
    assert loaded_yaml["service"]["name"] == "mistral-7b-q4km"
    assert loaded_yaml["service"]["parameters"]["context_size"] == 4096

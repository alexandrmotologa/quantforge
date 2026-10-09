"""Tests for Typer CLI commands."""

from typer.testing import CliRunner
from pathlib import Path
from quantforge.cli.app import cli
from .helpers import write_synthetic_gguf

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "QuantForge version" in result.stdout


def test_cli_bin_paths():
    result = runner.invoke(cli, ["bin", "paths"])
    assert result.exit_code == 0
    assert "Configured Search Directories:" in result.stdout


def test_cli_bin_check():
    result = runner.invoke(cli, ["bin", "check"])
    assert result.exit_code == 0
    assert "llama.cpp Binary Status" in result.stdout


def test_cli_pipeline_init(tmp_path: Path):
    out_recipe = tmp_path / "test_recipe.yaml"
    result = runner.invoke(cli, ["pipeline", "init", "--output", str(out_recipe)])
    assert result.exit_code == 0
    assert out_recipe.is_file()
    assert "Recipe initialized at:" in result.stdout


def test_cli_inspect(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    result = runner.invoke(cli, ["inspect", str(model)])
    assert result.exit_code == 0
    assert "GGUF Model Info:" in result.stdout
    assert "Architecture" in result.stdout
    assert "llama" in result.stdout

    # Test json output
    json_res = runner.invoke(cli, ["inspect", str(model), "--json"])
    assert json_res.exit_code == 0
    assert '"architecture": "llama"' in json_res.stdout


def test_cli_pull_help():
    res = runner.invoke(cli, ["pull", "--help"])
    assert res.exit_code == 0
    assert "Download a GGUF checkpoint" in res.stdout


def test_cli_push_help():
    res = runner.invoke(cli, ["push", "--help"])
    assert res.exit_code == 0
    assert "Publish quantized models" in res.stdout


def test_cli_vram_calc(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    res = runner.invoke(cli, ["vram-calc", str(model), "--ctx-size", "4096", "--kv-quant", "q8_0"])
    assert res.exit_code == 0
    assert "VRAM Breakdown" in res.stdout
    assert "GPU Hardware Compatibility Matrix" in res.stdout
    assert "FITS" in res.stdout


def test_cli_export_ollama_and_inferops(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    modelfile = tmp_path / "Modelfile"
    res_ollama = runner.invoke(cli, ["export", "ollama", str(model), "--output", str(modelfile)])
    assert res_ollama.exit_code == 0
    assert modelfile.is_file()
    assert "Ollama Modelfile successfully written to" in res_ollama.stdout

    inferops_yaml = tmp_path / "inferops.yaml"
    res_inferops = runner.invoke(cli, ["export", "inferops", str(model), "--output", str(inferops_yaml)])
    assert res_inferops.exit_code == 0
    assert inferops_yaml.is_file()
    assert "InferOps manifest successfully written to" in res_inferops.stdout


def test_cli_split_and_merge_help():
    res_split = runner.invoke(cli, ["split", "--help"])
    assert res_split.exit_code == 0
    assert "Split a multi-gigabyte GGUF model" in res_split.stdout

    res_merge = runner.invoke(cli, ["merge-shards", "--help"])
    assert res_merge.exit_code == 0
    assert "Reassemble GGUF sharded volumes" in res_merge.stdout


def test_cli_lora_and_canary_help():
    res_lora = runner.invoke(cli, ["lora", "merge", "--help"])
    assert res_lora.exit_code == 0
    assert "Merge LoRA adapter weights" in res_lora.stdout

    res_canary = runner.invoke(cli, ["canary", "--help"])
    assert res_canary.exit_code == 0
    assert "zero-shot canary prompts" in res_canary.stdout


def test_cli_report(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    html_out = tmp_path / "report.html"
    res = runner.invoke(cli, ["report", str(tmp_path), "--output", str(html_out)])
    assert res.exit_code == 0
    assert html_out.is_file()
    assert "Standalone HTML benchmark report generated at" in res.stdout

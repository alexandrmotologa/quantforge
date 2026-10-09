"""Tests for FastAPI Web application and endpoints."""

from pathlib import Path
from fastapi.testclient import TestClient
from quantforge.web.app import app
from .helpers import write_synthetic_gguf

client = TestClient(app)


def test_api_system_status():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "binaries" in data
    assert "cpu_count" in data


def test_api_jobs_list():
    response = client.get("/api/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_eval_pareto():
    points = [
        {"quant_type": "F16", "file_size_gb": 14.0, "perplexity": 5.0, "is_baseline": True},
        {"quant_type": "Q4_K_M", "file_size_gb": 4.1, "perplexity": 5.15},
    ]
    response = client.post("/api/eval/pareto", json={"points": points, "baseline_quant": "F16"})
    assert response.status_code == 200
    data = response.json()
    assert data["baseline_quant"] == "F16"
    assert "Q4_K_M" in data["pareto_frontier_quants"]


def test_api_model_inspect(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    response = client.post("/api/models/inspect", json={"model_path": str(model)})
    assert response.status_code == 200
    data = response.json()
    assert data["architecture"] == "llama"
    assert data["context_length"] == 4096


def test_web_dashboard_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "QuantForge" in response.text
    assert "Model Optimization Lab" in response.text


def test_web_studio_html():
    response = client.get("/studio")
    assert response.status_code == 200
    assert "Quantization Studio" in response.text


def test_web_eval_html():
    response = client.get("/eval")
    assert response.status_code == 200
    assert "Pareto Quality Lab" in response.text


def test_web_pipelines_html():
    response = client.get("/pipelines")
    assert response.status_code == 200
    assert "Automated Pipelines Lab" in response.text


def test_web_models_html():
    response = client.get("/models")
    assert response.status_code == 200
    assert "Models Library" in response.text


def test_api_autotune():
    response = client.get("/api/system/autotune?model_size_gb=4.0")
    assert response.status_code == 200
    data = response.json()
    assert "optimal_threads" in data
    assert "recommended_n_gpu_layers" in data


def test_api_datasets_presets():
    response = client.get("/api/datasets/presets")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 4
    assert any(p["name"] == "general-wiki" for p in data)


def test_api_pipeline_run_dispatch():
    payload = {
        "input_model": "test_model.gguf",
        "output_dir": "./dist",
        "corpus_preset": "general-wiki",
        "quants": ["Q4_K_M"],
        "evaluate": False,
        "export_card": False,
    }
    response = client.post("/api/pipelines/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_api_vram_calculate(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    payload = {
        "model_path": str(model),
        "context_size": 4096,
        "kv_quant": "q8_0",
    }
    response = client.post("/api/vram/calculate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "total_vram_gb" in data
    assert "fits_gpus" in data
    assert "8GB (RTX 4060 / Apple M-series 8G)" in data["fits_gpus"]


def test_api_export_endpoints(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    res_ollama = client.post("/api/export", json={"model_path": str(model), "format": "ollama"})
    assert res_ollama.status_code == 200
    assert "FROM " in res_ollama.json()["content"]

    res_infer = client.post("/api/export", json={"model_path": str(model), "format": "inferops"})
    assert res_infer.status_code == 200
    assert "llama.cpp" in res_infer.json()["content"]


def test_web_report_html_view(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    response = client.get(f"/report/html?models_dir={str(tmp_path)}")
    assert response.status_code == 200
    assert "QuantForge" in response.text
    assert "Optimization Report" in response.text
    assert "Pareto Quality vs. Size Frontier" in response.text


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

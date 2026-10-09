"""Tests for HTML report generator."""

from pathlib import Path
import pytest

from quantforge.engines.pareto import ParetoReport, QuantPoint
from quantforge.formats.html_report import HTMLReportGenerator


def test_html_report_generation(tmp_path: Path):
    points = [
        QuantPoint(
            file_path=Path("m-fp16.gguf"),
            quant_type="FP16",
            file_size_gb=14.0,
            perplexity=5.00,
            delta_ppl=0.0,
            quality_score=100.0,
            is_pareto_optimal=True,
        ),
        QuantPoint(
            file_path=Path("m-q4km.gguf"),
            quant_type="Q4_K_M",
            file_size_gb=4.1,
            perplexity=5.12,
            delta_ppl=0.12,
            quality_score=95.0,
            is_pareto_optimal=True,
        ),
        QuantPoint(
            file_path=Path("m-iq2xxs.gguf"),
            quant_type="IQ2_XXS",
            file_size_gb=2.2,
            perplexity=6.80,
            delta_ppl=1.80,
            quality_score=68.0,
            is_pareto_optimal=False,
        ),
    ]

    report = ParetoReport(
        points=points,
        pareto_frontier=[points[0], points[1]],
        baseline_ppl=5.00,
        baseline_quant="FP16",
        baseline_size_gb=14.0,
    )

    out_file = tmp_path / "report.html"
    generator = HTMLReportGenerator()
    html = generator.generate(
        report=report,
        model_name="Mistral-7B-v0.3",
        hardware_info="NVIDIA RTX 4090 24GB",
        output_file=out_file,
    )

    assert "<!DOCTYPE html>" in html
    assert "Mistral-7B-v0.3" in html
    assert "NVIDIA RTX 4090 24GB" in html
    assert "PARETO OPTIMAL" in html
    assert "DOMINATED" in html
    assert "svg" in html.lower()
    assert "updateSimulator" in html
    assert out_file.is_file()
    assert len(out_file.read_text(encoding="utf-8")) > 1000

"""Tests for ParetoSvgGenerator."""

from pathlib import Path
from quantforge.engines.pareto import ParetoAnalyzer, QuantPoint
from quantforge.formats.pareto_svg import ParetoSvgGenerator


def test_pareto_svg_generation(tmp_path: Path):
    generator = ParetoSvgGenerator()

    pts = [
        QuantPoint(quant_type="F16", file_path=Path("f16.gguf"), file_size_gb=14.0, perplexity=5.00, is_baseline=True),
        QuantPoint(quant_type="Q8_0", file_path=Path("q80.gguf"), file_size_gb=7.7, perplexity=5.01),
        QuantPoint(quant_type="Q4_K_M", file_path=Path("q4km.gguf"), file_size_gb=4.1, perplexity=5.15),
        QuantPoint(quant_type="IQ3_M", file_path=Path("iq3m.gguf"), file_size_gb=3.3, perplexity=5.35),
    ]

    report = ParetoAnalyzer().compute_frontier(pts)
    svg_str = generator.generate_svg(report)

    assert "<svg" in svg_str
    assert "</svg>" in svg_str
    assert "QuantForge Pareto Quality Frontier" in svg_str
    assert "Q4_K_M" in svg_str
    assert "polyline" in svg_str

    # Test saving to disk
    out_svg = tmp_path / "chart.svg"
    saved = generator.save_svg(report, out_svg)
    assert saved.is_file()
    assert len(saved.read_text(encoding="utf-8")) > 500


def test_pareto_svg_empty_report():
    generator = ParetoSvgGenerator()
    report = ParetoAnalyzer().compute_frontier([])
    svg_str = generator.generate_svg(report)
    assert "No points to display" in svg_str

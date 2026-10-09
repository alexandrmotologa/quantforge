"""Tests for ParetoAnalyzer trade-off optimization."""

from pathlib import Path
from quantforge.engines.pareto import ParetoAnalyzer, QuantPoint


def test_pareto_frontier_identification():
    analyzer = ParetoAnalyzer()

    # Define candidate points (Size GB, PPL)
    # F16: 14GB, 5.0 (Baseline)
    # Q8_0: 7.7GB, 5.01
    # Q4_K_M: 4.1GB, 5.15
    # Sub-optimal: 4.5GB, 5.30 (dominated by Q4_K_M which is both smaller and lower PPL)
    # IQ3_M: 3.3GB, 5.35
    # IQ2_M: 2.5GB, 6.10

    pts = [
        QuantPoint(quant_type="F16", file_path=Path("f16.gguf"), file_size_gb=14.0, perplexity=5.00, is_baseline=True),
        QuantPoint(quant_type="Q8_0", file_path=Path("q80.gguf"), file_size_gb=7.7, perplexity=5.01),
        QuantPoint(quant_type="Q4_K_M", file_path=Path("q4km.gguf"), file_size_gb=4.1, perplexity=5.15),
        QuantPoint(quant_type="DOMINATED_4BIT", file_path=Path("dom.gguf"), file_size_gb=4.5, perplexity=5.30),
        QuantPoint(quant_type="IQ3_M", file_path=Path("iq3m.gguf"), file_size_gb=3.3, perplexity=5.35),
        QuantPoint(quant_type="IQ2_M", file_path=Path("iq2m.gguf"), file_size_gb=2.5, perplexity=6.10),
    ]

    report = analyzer.compute_frontier(pts)

    frontier_quants = [p.quant_type for p in report.pareto_frontier]
    assert "Q4_K_M" in frontier_quants
    assert "IQ3_M" in frontier_quants
    assert "IQ2_M" in frontier_quants
    assert "DOMINATED_4BIT" not in frontier_quants

    # Check serialization
    d = report.to_dict()
    assert d["baseline_quant"] == "F16"
    assert "Q4_K_M" in d["pareto_frontier_quants"]

    # Check Markdown rendering
    md = report.to_markdown_table()
    assert "| `Q4_K_M` | 4.10 |" in md
    assert "Yes" in md

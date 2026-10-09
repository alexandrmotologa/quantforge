"""Pareto frontier analyzer computing size, perplexity, and speed efficiency."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QuantPoint:
    """Evaluation point for a specific quantized checkpoint."""

    quant_type: str
    file_path: Path
    file_size_gb: float
    perplexity: float
    speed_tok_s: Optional[float] = None
    delta_ppl: Optional[float] = None
    is_baseline: bool = False
    is_pareto_optimal: bool = False
    quality_score: float = 0.0
    recommendation: Optional[str] = None


@dataclass
class ParetoReport:
    """Consolidated Pareto frontier report."""

    baseline_quant: Optional[str] = None
    baseline_ppl: Optional[float] = None
    baseline_size_gb: Optional[float] = None
    points: List[QuantPoint] = field(default_factory=list)
    pareto_frontier: List[QuantPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to dictionary."""
        return {
            "baseline_quant": self.baseline_quant,
            "baseline_ppl": self.baseline_ppl,
            "baseline_size_gb": self.baseline_size_gb,
            "points": [
                {
                    "quant_type": p.quant_type,
                    "file_size_gb": round(p.file_size_gb, 3),
                    "perplexity": round(p.perplexity, 4),
                    "delta_ppl": round(p.delta_ppl, 4) if p.delta_ppl is not None else None,
                    "speed_tok_s": p.speed_tok_s,
                    "is_pareto_optimal": p.is_pareto_optimal,
                    "quality_score": round(p.quality_score, 2),
                    "recommendation": p.recommendation,
                }
                for p in self.points
            ],
            "pareto_frontier_quants": [p.quant_type for p in self.pareto_frontier],
        }

    def to_markdown_table(self) -> str:
        """Renders report as a clean Markdown table."""
        lines = [
            "| Quant Type | Size (GB) | Perplexity | Delta PPL | Pareto Optimal | Recommendation |",
            "|---|---|---|---|---|---|",
        ]
        for p in sorted(self.points, key=lambda x: x.file_size_gb):
            star = "Yes" if p.is_pareto_optimal else "No"
            delta = f"+{p.delta_ppl:.4f}" if p.delta_ppl and p.delta_ppl > 0 else (f"{p.delta_ppl:.4f}" if p.delta_ppl is not None else "-")
            rec = p.recommendation or "-"
            lines.append(
                f"| `{p.quant_type}` | {p.file_size_gb:.2f} | {p.perplexity:.4f} | {delta} | {star} | {rec} |"
            )
        return "\n".join(lines)


class ParetoAnalyzer:
    """Calculates non-dominated Pareto frontier points for model selections."""

    def compute_frontier(
        self,
        points: List[QuantPoint],
        baseline_quant: Optional[str] = None,
    ) -> ParetoReport:
        """Determines Pareto-optimal quants minimizing Size and Perplexity."""
        if not points:
            return ParetoReport()

        # Identify baseline
        baseline_pt = None
        for p in points:
            if p.is_baseline or (baseline_quant and p.quant_type.upper() == baseline_quant.upper()):
                baseline_pt = p
                p.is_baseline = True
                break

        base_ppl = baseline_pt.perplexity if baseline_pt else None
        base_size = baseline_pt.file_size_gb if baseline_pt else None

        for p in points:
            if base_ppl is not None and not p.is_baseline:
                p.delta_ppl = round(p.perplexity - base_ppl, 4)

        # Pareto frontier: Minimizing (Size, PPL)
        # Point A dominates B if Size(A) <= Size(B) and PPL(A) <= PPL(B) with at least one strict
        pareto_list: List[QuantPoint] = []
        for p1 in points:
            dominated = False
            for p2 in points:
                if p1 is p2:
                    continue
                # If p2 is smaller or equal in size AND has lower or equal perplexity
                if p2.file_size_gb <= p1.file_size_gb and p2.perplexity <= p1.perplexity:
                    if p2.file_size_gb < p1.file_size_gb or p2.perplexity < p1.perplexity:
                        dominated = True
                        break
            if not dominated:
                p1.is_pareto_optimal = True
                pareto_list.append(p1)
            else:
                p1.is_pareto_optimal = False

        # Quality scoring (Higher is better: lower size penalty, lower PPL penalty)
        # Score = 100 / ( (size_gb / min_size) * (ppl / min_ppl) )
        min_size = min(p.file_size_gb for p in points)
        min_ppl = min(p.perplexity for p in points)

        for p in points:
            size_factor = p.file_size_gb / max(min_size, 0.001)
            ppl_factor = p.perplexity / max(min_ppl, 0.001)
            p.quality_score = 100.0 / (size_factor * (ppl_factor ** 1.5))

            # Assign recommendations
            q_upper = p.quant_type.upper()
            if p.is_pareto_optimal:
                if "Q4_K_M" in q_upper or "IQ4_XS" in q_upper:
                    p.recommendation = "Daily Driver"
                elif "IQ3_M" in q_upper or "Q3_K_M" in q_upper:
                    p.recommendation = "Best Compact"
                elif "IQ2" in q_upper or "Q2_K" in q_upper:
                    p.recommendation = "Ultra Memory Saver"
                elif "Q6_K" in q_upper or "Q8_0" in q_upper:
                    p.recommendation = "Lossless Alternative"

        # Sort pareto frontier by size
        pareto_list.sort(key=lambda x: x.file_size_gb)

        return ParetoReport(
            baseline_quant=baseline_pt.quant_type if baseline_pt else None,
            baseline_ppl=base_ppl,
            baseline_size_gb=base_size,
            points=points,
            pareto_frontier=pareto_list,
        )

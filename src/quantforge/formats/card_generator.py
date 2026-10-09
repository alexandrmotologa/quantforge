"""Model Card and publication documentation generator for quantized GGUF models."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from quantforge.engines.pareto import ParetoReport, QuantPoint
from quantforge.formats.gguf_reader import GGUFModelInfo


class ModelCardGenerator:
    """Generates standard Markdown model cards with benchmark and VRAM tables."""

    @staticmethod
    def estimate_vram_requirement_gb(file_size_gb: float, ctx_length: int = 4096) -> float:
        """Estimates required VRAM including KV cache overhead."""
        # KV cache overhead approximation: ~0.5GB per 4k tokens for 7B-class models
        kv_overhead = (ctx_length / 4096.0) * 0.75
        total_vram = file_size_gb + kv_overhead + 0.5  # Context buffers + scratchpad
        return round(total_vram, 1)

    def generate_card(
        self,
        model_name: str,
        model_info: Optional[GGUFModelInfo],
        pareto_report: Optional[ParetoReport] = None,
        quant_points: Optional[List[QuantPoint]] = None,
    ) -> str:
        """Constructs full Markdown document."""
        arch = model_info.architecture if model_info else "Transformer"
        ctx = model_info.context_length if model_info and model_info.context_length else 4096
        params = f"{model_info.estimated_parameters / 1e9:.1f}B" if model_info and model_info.estimated_parameters else "N/A"

        lines = [
            f"# {model_name} - GGUF Quantizations",
            "",
            f"Quantized and calibrated using **QuantForge**.",
            "",
            "## Architecture Specifications",
            "",
            f"- **Base Architecture**: {arch}",
            f"- **Parameters**: {params}",
            f"- **Context Length**: {ctx} tokens",
            "",
            "## Quantization Overview & Benchmarks",
            "",
        ]

        if pareto_report and pareto_report.points:
            lines.append("![Pareto Quality Frontier](pareto_frontier.svg)")
            lines.append("")
            lines.append(pareto_report.to_markdown_table())
            lines.append("")
        elif quant_points:
            lines.extend([
                "| Quant | Size (GB) | Est. VRAM (GB) |",
                "|---|---|---|",
            ])
            for q in sorted(quant_points, key=lambda x: x.file_size_gb):
                vram = self.estimate_vram_requirement_gb(q.file_size_gb, ctx)
                lines.append(f"| `{q.quant_type}` | {q.file_size_gb:.2f} GB | {vram} GB |")
            lines.append("")

        lines.extend([
            "## Recommended Quants",
            "",
            "- **Daily Driver**: `Q4_K_M` or `IQ4_XS` (best general balance between retention and speed).",
            "- **Resource-Constrained**: `IQ3_M` (compact footprint with minimal perplexity degradation).",
            "- **Near-Lossless**: `Q6_K` or `Q8_0` (for high-fidelity reasoning and coding tasks).",
            "",
            "## Inference Examples",
            "",
            "### Using llama.cpp",
            "",
            "```bash",
            f"llama-cli -m {model_name}-Q4_K_M.gguf -p \"Hello! How can I assist you today?\" -c {ctx} -ngl 33",
            "```",
            "",
            "### Using Ollama",
            "",
            "Create a `Modelfile`:",
            "",
            "```dockerfile",
            f"FROM ./{model_name}-Q4_K_M.gguf",
            "TEMPLATE \"\"\"{{ .Prompt }}\"\"\"",
            "```",
            "",
            "Then build and execute:",
            "",
            "```bash",
            f"ollama create {model_name.lower()} -f Modelfile",
            f"ollama run {model_name.lower()}",
            "```",
            "",
            "---",
            "*Generated automatically by QuantForge.*",
        ])

        return "\n".join(lines)

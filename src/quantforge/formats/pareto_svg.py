"""Procedural SVG generator for Pareto frontier quality versus size charts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from quantforge.engines.pareto import ParetoReport, QuantPoint


class ParetoSvgGenerator:
    """Renders standalone responsive SVG charts illustrating the Pareto frontier."""

    def __init__(self, width: int = 800, height: int = 450) -> None:
        self.width = width
        self.height = height
        self.padding_left = 70
        self.padding_right = 50
        self.padding_top = 50
        self.padding_bottom = 60

    def generate_svg(self, report: ParetoReport) -> str:
        """Constructs vector XML string for the given Pareto report."""
        points = report.points
        if not points:
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}"><text x="50%" y="50%" fill="#9ca3af" text-anchor="middle">No points to display</text></svg>'

        sizes = [p.file_size_gb for p in points]
        ppls = [p.perplexity for p in points]

        min_x = min(sizes) * 0.9
        max_x = max(sizes) * 1.05
        if max_x == min_x:
            max_x += 1.0

        min_y = min(ppls) * 0.98
        max_y = max(ppls) * 1.05
        if max_y == min_y:
            max_y += 1.0

        plot_w = self.width - self.padding_left - self.padding_right
        plot_h = self.height - self.padding_top - self.padding_bottom

        def to_x(val: float) -> float:
            return self.padding_left + ((val - min_x) / (max_x - min_x)) * plot_w

        def to_y(val: float) -> float:
            return (self.height - self.padding_bottom) - ((val - min_y) / (max_y - min_y)) * plot_h

        # Sort frontier for line rendering
        frontier = [p for p in points if p.is_pareto_optimal]
        frontier.sort(key=lambda p: p.file_size_gb)

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="100%" height="100%" style="background:#090d16;font-family:system-ui,-apple-system,sans-serif;">',
            '  <defs>',
            '    <linearGradient id="gridGrad" x1="0" y1="0" x2="0" y2="1">',
            '      <stop offset="0%" stop-color="#1e293b" stop-opacity="0.2"/>',
            '      <stop offset="100%" stop-color="#0f172a" stop-opacity="0.8"/>',
            '    </linearGradient>',
            '  </defs>',
            f'  <rect width="{self.width}" height="{self.height}" rx="12" fill="url(#gridGrad)" stroke="#1e293b" stroke-width="1"/>',
            f'  <text x="{self.width // 2}" y="30" fill="#f8fafc" font-size="16" font-weight="bold" text-anchor="middle">QuantForge Pareto Quality Frontier</text>',
            f'  <text x="{self.width // 2}" y="{self.height - 18}" fill="#94a3b8" font-size="12" text-anchor="middle">Model Size (GB) →</text>',
            f'  <text x="22" y="{self.height // 2}" fill="#94a3b8" font-size="12" transform="rotate(-90 22 {self.height // 2})" text-anchor="middle">Perplexity (Lower is better) →</text>',
        ]

        # Draw gridlines & axis ticks
        x_ticks = 5
        for i in range(x_ticks + 1):
            val_x = min_x + (i / x_ticks) * (max_x - min_x)
            px = to_x(val_x)
            svg.append(f'  <line x1="{px:.1f}" y1="{self.padding_top}" x2="{px:.1f}" y2="{self.height - self.padding_bottom}" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>')
            svg.append(f'  <text x="{px:.1f}" y="{self.height - self.padding_bottom + 16}" fill="#64748b" font-size="10" text-anchor="middle">{val_x:.1f}</text>')

        y_ticks = 5
        for i in range(y_ticks + 1):
            val_y = min_y + (i / y_ticks) * (max_y - min_y)
            py = to_y(val_y)
            svg.append(f'  <line x1="{self.padding_left}" y1="{py:.1f}" x2="{self.width - self.padding_right}" y2="{py:.1f}" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>')
            svg.append(f'  <text x="{self.padding_left - 8}" y="{py + 3:.1f}" fill="#64748b" font-size="10" text-anchor="end">{val_y:.2f}</text>')

        # Draw Pareto frontier polyline
        if len(frontier) > 1:
            points_str = " ".join(f"{to_x(p.file_size_gb):.1f},{to_y(p.perplexity):.1f}" for p in frontier)
            svg.append(f'  <polyline points="{points_str}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="6 4"/>')

        # Draw points and labels
        for p in points:
            cx = to_x(p.file_size_gb)
            cy = to_y(p.perplexity)
            color = "#10b981" if p.is_pareto_optimal else "#64748b"
            r = 6 if p.is_pareto_optimal else 4

            svg.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" stroke="#0f172a" stroke-width="2"/>')
            # Text label
            lbl_color = "#34d399" if p.is_pareto_optimal else "#94a3b8"
            weight = "bold" if p.is_pareto_optimal else "normal"
            svg.append(f'  <text x="{cx + 8:.1f}" y="{cy - 5:.1f}" fill="{lbl_color}" font-size="11" font-weight="{weight}">{p.quant_type}</text>')

        # Legend
        leg_x = self.width - self.padding_right - 140
        leg_y = self.padding_top + 10
        svg.extend([
            f'  <rect x="{leg_x}" y="{leg_y}" width="135" height="52" rx="6" fill="#0f172a" stroke="#334155" stroke-width="1" opacity="0.9"/>',
            f'  <circle cx="{leg_x + 15}" cy="{leg_y + 18}" r="5" fill="#10b981"/>',
            f'  <text x="{leg_x + 28}" y="{leg_y + 22}" fill="#e2e8f0" font-size="11">Pareto Optimal</text>',
            f'  <circle cx="{leg_x + 15}" cy="{leg_y + 36}" r="4" fill="#64748b"/>',
            f'  <text x="{leg_x + 28}" y="{leg_y + 40}" fill="#94a3b8" font-size="11">Sub-optimal</text>',
            '</svg>',
        ])

        return "\n".join(svg)

    def save_svg(self, report: ParetoReport, target_path: Path) -> Path:
        """Generates and writes SVG file to disk."""
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        svg_content = self.generate_svg(report)
        target.write_text(svg_content, encoding="utf-8")
        return target

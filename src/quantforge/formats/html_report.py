"""Standalone interactive HTML report generator with embedded Pareto SVG and VRAM calculator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from quantforge.engines.pareto import ParetoReport
from quantforge.formats.pareto_svg import ParetoSvgGenerator


class HTMLReportGenerator:
    """Produces a zero-dependency, self-contained offline HTML benchmark report."""

    def __init__(self, svg_generator: Optional[ParetoSvgGenerator] = None) -> None:
        self.svg_generator = svg_generator or ParetoSvgGenerator(width=850, height=420)

    def generate(
        self,
        report: ParetoReport,
        model_name: str,
        hardware_info: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """Constructs standalone HTML document with embedded CSS, SVG chart, and interactive JS."""
        svg_markup = self.svg_generator.generate_svg(report)

        # Prepare JSON payload for the interactive client-side VRAM simulator
        points_data = [
            {
                "quant": p.quant_type,
                "size_gb": p.file_size_gb,
                "size_bytes": int(p.file_size_gb * 1024 * 1024 * 1024),
                "ppl": round(p.perplexity, 4),
                "delta_ppl": round(p.delta_ppl, 4) if p.delta_ppl is not None else 0.0,
                "score": round(p.quality_score, 1),
                "is_optimal": p.is_pareto_optimal,
            }
            for p in report.points
        ]
        points_json = json.dumps(points_data)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # HTML template with embedded dark-mode styles and zero CDN dependencies
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QuantForge Benchmark Report - {model_name}</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #0f172a;
      --border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 2.5rem 1.5rem;
    }}
    .container {{
      max-width: 1000px;
      margin: 0 auto;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--primary);
    }}
    .brand-pill {{
      font-size: 0.75rem;
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}
    .meta-text {{
      color: var(--text-muted);
      font-size: 0.875rem;
      margin-top: 0.25rem;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
    .card-title {{
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.9rem;
    }}
    th {{
      color: var(--text-muted);
      font-weight: 600;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 0.75rem 1rem;
      border-bottom: 1px solid rgba(30, 41, 59, 0.5);
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-optimal {{ background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-suboptimal {{ background: rgba(148, 163, 184, 0.1); color: var(--text-muted); border: 1px solid var(--border); }}
    .badge-fit {{ background: rgba(16, 185, 129, 0.2); color: var(--success); }}
    .badge-oom {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
    .control-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}
    label {{
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
    }}
    input[type=range] {{
      width: 100%;
      accent-color: var(--primary);
    }}
    .radio-group {{
      display: flex;
      gap: 1rem;
    }}
    .radio-label {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.875rem;
      cursor: pointer;
    }}
    .vram-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }}
    .gpu-card {{
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1rem;
    }}
    .gpu-title {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 0.25rem;
    }}
    .gpu-status {{
      font-size: 0.95rem;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="brand">
          QuantForge
          <span class="brand-pill">Optimization Report</span>
        </div>
        <div class="meta-text">Model: <strong>{model_name}</strong> | Generated: {timestamp}</div>
        {f'<div class="meta-text">Hardware: {hardware_info}</div>' if hardware_info else ''}
      </div>
    </div>

    <!-- Pareto Frontier SVG Chart Card -->
    <div class="card">
      <div class="card-title">
        <span>Pareto Quality vs. Size Frontier</span>
      </div>
      <div style="width: 100%; overflow-x: auto;">
        {svg_markup}
      </div>
    </div>

    <!-- Quantization Benchmark Table -->
    <div class="card">
      <div class="card-title">
        <span>Quantization Benchmark Table</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Quantization Type</th>
            <th>File Size (GB)</th>
            <th>Perplexity (PPL)</th>
            <th>Δ PPL</th>
            <th>Quality Score</th>
            <th>Frontier Status</th>
          </tr>
        </thead>
        <tbody>
          {"".join([
              f'''<tr>
                <td><strong>{p.quant_type}</strong></td>
                <td>{p.file_size_gb:.2f} GB</td>
                <td>{p.perplexity:.4f}</td>
                <td style="color: {'#10b981' if (p.delta_ppl or 0) <= 0.2 else '#f59e0b' if (p.delta_ppl or 0) <= 0.8 else '#ef4444'}">{f"+{p.delta_ppl:.4f}" if p.delta_ppl is not None and p.delta_ppl > 0 else (f"{p.delta_ppl:.4f}" if p.delta_ppl is not None else "-")}</td>
                <td><strong>{p.quality_score:.1f}</strong> / 100</td>
                <td><span class="badge {'badge-optimal' if p.is_pareto_optimal else 'badge-suboptimal'}">{'PARETO OPTIMAL' if p.is_pareto_optimal else 'DOMINATED'}</span></td>
              </tr>'''
              for p in report.points
          ])}
        </tbody>
      </table>
    </div>

    <!-- Interactive VRAM & KV Cache Simulator -->
    <div class="card">
      <div class="card-title">
        <span>Interactive VRAM & Context Length Simulator</span>
      </div>
      <div class="control-row">
        <div class="control-group">
          <label id="ctxLabel">Context Window: <strong>8192 tokens</strong></label>
          <input type="range" id="ctxSlider" min="2048" max="131072" step="2048" value="8192" oninput="updateSimulator()">
        </div>
        <div class="control-group">
          <label>KV Cache Quantization:</label>
          <div class="radio-group">
            <label class="radio-label"><input type="radio" name="kvQuant" value="fp16" checked onchange="updateSimulator()"> FP16 (2.0B)</label>
            <label class="radio-label"><input type="radio" name="kvQuant" value="q8_0" onchange="updateSimulator()"> Q8_0 (1.06B)</label>
            <label class="radio-label"><input type="radio" name="kvQuant" value="q4_0" onchange="updateSimulator()"> Q4_0 (0.56B)</label>
          </div>
        </div>
      </div>

      <div class="vram-grid" id="gpuGrid">
        <!-- Rendered dynamically by JavaScript -->
      </div>
    </div>
  </div>

  <script>
    const quants = {points_json};
    const gpuTiers = [
      {{ name: "8GB (RTX 4060)", limit: 8.0 }},
      {{ name: "12GB (RTX 4070)", limit: 12.0 }},
      {{ name: "16GB (RTX 4080)", limit: 16.0 }},
      {{ name: "24GB (RTX 3090/4090)", limit: 24.0 }},
      {{ name: "48GB (A6000 Ada)", limit: 48.0 }},
      {{ name: "80GB (A100/H100)", limit: 80.0 }}
    ];

    function updateSimulator() {{
      const ctx = parseInt(document.getElementById('ctxSlider').value);
      document.getElementById('ctxLabel').innerHTML = `Context Window: <strong>${{ctx.toLocaleString()}} tokens</strong>`;
      
      const kvQuant = document.querySelector('input[name="kvQuant"]:checked').value;
      const bpe = kvQuant === "fp16" ? 2.0 : (kvQuant === "q8_0" ? 1.0625 : 0.5625);

      // KV cache estimate for standard 32 layers, 8 KV heads, 128 head dim
      const kvBytes = 2 * 32 * 8 * 128 * ctx * bpe;
      const kvGb = kvBytes / (1024 * 1024 * 1024);
      const overheadGb = 0.6; // scratch + cuda

      // Pick top optimal quant for demonstration
      const topQuant = quants.find(q => q.is_optimal) || quants[0];
      const modelGb = topQuant ? topQuant.size_gb : 4.0;
      const totalGb = modelGb + kvGb + overheadGb;

      const container = document.getElementById('gpuGrid');
      container.innerHTML = gpuTiers.map(gpu => {{
        const fits = totalGb <= gpu.limit;
        const remaining = (gpu.limit - totalGb).toFixed(1);
        return `
          <div class="gpu-card">
            <div class="gpu-title">${{gpu.name}}</div>
            <div class="gpu-status">
              <span class="badge ${{fits ? 'badge-fit' : 'badge-oom'}}">${{fits ? 'FITS' : 'OOM'}}</span>
              <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.4rem;">
                ${{fits ? remaining + ' GB free' : 'Need ' + totalGb.toFixed(1) + ' GB'}}
              </span>
            </div>
          </div>
        `;
      }}).join('');
    }}

    updateSimulator();
  </script>
</body>
</html>
"""

        if output_file:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")

        return html

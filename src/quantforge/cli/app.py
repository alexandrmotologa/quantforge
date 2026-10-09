"""Command-line interface for QuantForge powered by Typer and Rich."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from quantforge import __version__
from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import (
    BenchmarkConfig,
    ImatrixConfig,
    MatrixQuantConfig,
    PerplexityConfig,
    QuantizationConfig,
    QuantType,
    settings,
)
from quantforge.engines.benchmark import BenchmarkEngine
from quantforge.engines.imatrix import ImatrixEngine
from quantforge.engines.pareto import ParetoAnalyzer, QuantPoint
from quantforge.engines.perplexity import PerplexityEngine
from quantforge.engines.quantize import QuantizeEngine
from quantforge.formats.gguf_reader import GGUFReader
from quantforge.pipelines.runner import PipelineRunner
from quantforge.pipelines.templates import BALANCED_RECIPE, EXTREME_COMPRESSION_RECIPE, dump_recipe_to_yaml

cli = typer.Typer(
    name="quantforge",
    help="QuantForge: Automated GGUF quantization, imatrix calibration, and perplexity evaluation lab.",
    no_args_is_help=True,
)
bin_cli = typer.Typer(help="Manage llama.cpp native binary discovery, inspection, and downloads.")
pipeline_cli = typer.Typer(help="Automate multi-stage quantization workflows via YAML recipes.")
dataset_cli = typer.Typer(help="Manage and prepare calibration datasets for importance matrices.")

cli.add_typer(bin_cli, name="bin")
cli.add_typer(pipeline_cli, name="pipeline")
cli.add_typer(dataset_cli, name="dataset")

console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]QuantForge[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@cli.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version and exit.", callback=version_callback, is_eager=True
    ),
):
    pass


# -------------------------------------------------------------------------
# BINARY MANAGEMENT COMMANDS
# -------------------------------------------------------------------------

@bin_cli.command("check")
def bin_check():
    """Verify presence and capabilities of required llama.cpp executables."""
    mgr = BinaryManager()
    status = mgr.get_status()

    table = Table(title="llama.cpp Binary Status", border_style="cyan")
    table.add_column("Tool", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Version / Build", style="dim")
    table.add_column("Resolved Path", style="cyan")

    for name, info in status.items():
        if info.found:
            table.add_row(name, "[green]Found[/green]", info.version or "available", str(info.path))
        else:
            table.add_row(name, "[red]Missing[/red]", "-", "[dim]" + (info.error or "Not found") + "[/dim]")

    console.print(table)


@bin_cli.command("download")
def bin_download(
    backend: str = typer.Option("cpu", "--backend", "-b", help="Target hardware: cpu, cuda, vulkan"),
    dir_path: Optional[Path] = typer.Option(None, "--dir", "-d", help="Custom target directory"),
):
    """Download official precompiled llama.cpp release from GitHub."""
    mgr = BinaryManager()
    console.print(f"[yellow]Downloading official llama.cpp binaries for backend '{backend}'...[/yellow]")
    with console.status("[bold green]Downloading and extracting archive..."):
        try:
            target = mgr.download_release_binaries(backend=backend, target_dir=dir_path)
            console.print(f"[bold green]Successfully installed binaries to:[/bold green] {target}")
        except Exception as e:
            console.print(f"[bold red]Download failed:[/bold red] {e}")
            raise typer.Exit(code=1)


@bin_cli.command("paths")
def bin_paths():
    """Display ordered binary search directories."""
    mgr = BinaryManager()
    paths = mgr.get_search_paths()
    console.print("[bold cyan]Configured Search Directories:[/bold cyan]")
    for p in paths:
        exists = "[green](exists)[/green]" if p.is_dir() else "[dim](missing)[/dim]"
        console.print(f"  • {p} {exists}")


# -------------------------------------------------------------------------
# QUANTIZATION COMMANDS
# -------------------------------------------------------------------------

@cli.command("quantize")
def quantize_cmd(
    input_model: Path = typer.Argument(..., help="Path to source unquantized GGUF model"),
    output_model: Path = typer.Argument(..., help="Path to output quantized GGUF file"),
    quant: str = typer.Option("Q4_K_M", "--quant", "-q", help="Target quantization type"),
    imatrix: Optional[Path] = typer.Option(None, "--imatrix", "-i", help="Path to .dat importance matrix"),
    leave_output_tensor: bool = typer.Option(False, "--leave-output-tensor", help="Keep output tensor unquantized"),
    pure: bool = typer.Option(False, "--pure", help="Quantize all tensors including norms"),
    threads: Optional[int] = typer.Option(None, "--threads", "-t", help="CPU thread count"),
):
    """Quantize a GGUF model into a target format."""
    try:
        q_type = QuantType(quant.upper())
    except ValueError:
        console.print(f"[red]Invalid quantization format:[/red] {quant}")
        raise typer.Exit(1)

    cfg = QuantizationConfig(
        input_path=input_model,
        output_path=output_model,
        quant_type=q_type,
        imatrix_path=imatrix,
        leave_output_tensor=leave_output_tensor,
        pure=pure,
        threads=threads,
    )

    engine = QuantizeEngine()
    console.print(f"[cyan]Quantizing[/cyan] [bold]{input_model.name}[/bold] -> [bold green]{q_type.value}[/bold green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Quantizing to {q_type.value}", total=100)

        def _on_prog(p):
            progress.update(task, completed=p.percent, description=f"[{p.current_index}/{p.total_count}] {p.current_item}")

        res = asyncio.run(engine.quantize(cfg, on_progress=_on_prog))

    if res.success:
        orig_mb = res.original_size_bytes / (1024 * 1024)
        quant_mb = res.quantized_size_bytes / (1024 * 1024)
        console.print(
            Panel(
                f"[bold green]Quantization Complete![/bold green]\n\n"
                f"• Output: [cyan]{res.output_path}[/cyan]\n"
                f"• Format: [bold]{res.quant_type.value}[/bold]\n"
                f"• Original Size: {orig_mb:.1f} MB\n"
                f"• Quantized Size: {quant_mb:.1f} MB (Compression: {res.compression_ratio:.2f}x)\n"
                f"• Elapsed: {res.duration_seconds:.1f}s",
                title="Success",
                border_style="green",
            )
        )
    else:
        console.print(f"[bold red]Quantization failed:[/bold red] {res.error_message}")
        raise typer.Exit(1)


@cli.command("matrix")
def matrix_cmd(
    input_model: Path = typer.Argument(..., help="Path to input unquantized GGUF model"),
    output_dir: Path = typer.Option(Path("./quants"), "--output-dir", "-o", help="Output directory"),
    quants: str = typer.Option("Q4_K_M,Q5_K_M,Q8_0", "--quants", help="Comma-separated quant types"),
    imatrix: Optional[Path] = typer.Option(None, "--imatrix", "-i", help="Importance matrix .dat file"),
    threads: Optional[int] = typer.Option(None, "--threads", "-t", help="CPU thread count"),
):
    """Run a batch matrix quantization for multiple target formats."""
    types_list = []
    for q_name in quants.split(","):
        q_clean = q_name.strip().upper()
        if q_clean:
            try:
                types_list.append(QuantType(q_clean))
            except ValueError:
                console.print(f"[yellow]Skipping unknown quant type:[/yellow] {q_clean}")

    if not types_list:
        console.print("[red]No valid quantization types provided.[/red]")
        raise typer.Exit(1)

    cfg = MatrixQuantConfig(
        input_path=input_model,
        output_dir=output_dir,
        quant_types=types_list,
        imatrix_path=imatrix,
        threads=threads,
    )

    engine = QuantizeEngine()
    console.print(f"[cyan]Executing batch matrix for {len(types_list)} quants...[/cyan]")

    table = Table(title="Matrix Quantization Summary", border_style="cyan")
    table.add_column("Quant", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Duration", justify="right")

    def _on_finish(r):
        if r.success:
            q_mb = r.quantized_size_bytes / (1024 * 1024)
            table.add_row(
                r.quant_type.value,
                "[green]Success[/green]",
                f"{q_mb:.1f}",
                f"{r.compression_ratio:.2f}x",
                f"{r.duration_seconds:.1f}s",
            )
        else:
            table.add_row(r.quant_type.value, "[red]Failed[/red]", "-", "-", f"{r.duration_seconds:.1f}s")

    results = asyncio.run(engine.quantize_matrix(cfg, on_item_finish=_on_finish))
    console.print(table)


# -------------------------------------------------------------------------
# IMATRIX CALIBRATION COMMANDS
# -------------------------------------------------------------------------

@cli.command("imatrix")
def imatrix_cmd(
    input_model: Path = typer.Argument(..., help="Path to base model"),
    data: Path = typer.Option(..., "--data", "-d", help="Calibration text file"),
    output: Path = typer.Option(..., "--output", "-o", help="Target .dat imatrix file"),
    ctx_size: int = typer.Option(2048, "--ctx-size", "-c", help="Context window size"),
    chunks: int = typer.Option(64, "--chunks", help="Number of chunks to process"),
    n_gpu_layers: int = typer.Option(0, "--n-gpu-layers", "-ngl", help="GPU offloaded layers"),
    threads: Optional[int] = typer.Option(None, "--threads", "-t", help="CPU thread count"),
):
    """Calibrate importance matrix using a reference corpus."""
    cfg = ImatrixConfig(
        input_path=input_model,
        data_path=data,
        output_path=output,
        ctx_size=ctx_size,
        chunks=chunks,
        n_gpu_layers=n_gpu_layers,
        threads=threads,
    )

    engine = ImatrixEngine()
    console.print(f"[cyan]Calibrating imatrix with corpus:[/cyan] {data.name}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Calibrating...", total=100)

        def _on_prog(p):
            progress.update(task, completed=p.percent, description=p.current_item)

        res = asyncio.run(engine.calibrate(cfg, on_progress=_on_prog))

    if res.success:
        mb = res.file_size_bytes / (1024 * 1024)
        console.print(
            Panel(
                f"[bold green]Imatrix generated successfully![/bold green]\n\n"
                f"• Output: [cyan]{res.output_path}[/cyan]\n"
                f"• Size: {mb:.2f} MB\n"
                f"• Chunks: {res.chunks_processed}\n"
                f"• Elapsed: {res.duration_seconds:.1f}s",
                title="Success",
                border_style="green",
            )
        )
    else:
        console.print(f"[bold red]Calibration failed:[/bold red] {res.error_message}")
        raise typer.Exit(1)


# -------------------------------------------------------------------------
# EVALUATION & PERPLEXITY
# -------------------------------------------------------------------------

@cli.command("eval")
def eval_cmd(
    model: Path = typer.Argument(..., help="Model path to evaluate"),
    data: Path = typer.Option(..., "--data", "-d", help="Test dataset path"),
    ctx_size: int = typer.Option(2048, "--ctx-size", "-c", help="Context size"),
    n_gpu_layers: int = typer.Option(0, "--n-gpu-layers", "-ngl", help="GPU offload layers"),
    baseline_ppl: Optional[float] = typer.Option(None, "--baseline", help="Baseline FP16 PPL for delta calculation"),
):
    """Calculate perplexity of a model against an evaluation corpus."""
    cfg = PerplexityConfig(
        model_path=model,
        data_path=data,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
    )

    engine = PerplexityEngine()
    console.print(f"[cyan]Evaluating perplexity for:[/cyan] {model.name}")

    with console.status("[bold green]Computing perplexity over dataset..."):
        res = asyncio.run(engine.evaluate(cfg, baseline_ppl=baseline_ppl))

    if res.success:
        delta_str = f"{res.delta_perplexity:+.4f} ({res.degradation_pct:+.2f}%)" if res.delta_perplexity is not None else "-"
        console.print(
            Panel(
                f"[bold green]Perplexity Benchmark Result[/bold green]\n\n"
                f"• Model: [cyan]{model.name}[/cyan]\n"
                f"• Perplexity: [bold]{res.perplexity:.4f}[/bold] ± {res.perplexity_stderr or 0.0:.4f}\n"
                f"• Delta PPL vs Base: {delta_str}\n"
                f"• Evaluation Time: {res.duration_seconds:.1f}s",
                title="Result",
                border_style="cyan",
            )
        )
    else:
        console.print(f"[bold red]Evaluation failed:[/bold red] {res.error_message}")
        raise typer.Exit(1)


@cli.command("bench")
def bench_cmd(
    model: Path = typer.Argument(..., help="Path to GGUF model"),
    prompt_tokens: int = typer.Option(512, "-p", help="Prompt tokens"),
    gen_tokens: int = typer.Option(128, "-n", help="Generation tokens"),
    n_gpu_layers: int = typer.Option(0, "-ngl", help="GPU layers"),
):
    """Measure inference speed with llama-bench."""
    cfg = BenchmarkConfig(
        model_path=model,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        n_gpu_layers=n_gpu_layers,
    )
    engine = BenchmarkEngine()
    console.print(f"[cyan]Benchmarking inference throughput for:[/cyan] {model.name}")

    with console.status("[bold green]Profiling throughput..."):
        res = asyncio.run(engine.benchmark(cfg))

    if res.success:
        console.print(
            Panel(
                f"[bold green]Throughput Benchmark Result[/bold green]\n\n"
                f"• Prompt Processing: [bold]{res.prompt_processing_tok_s or 0.0:.2f} tok/s[/bold]\n"
                f"• Text Generation:   [bold]{res.text_generation_tok_s or 0.0:.2f} tok/s[/bold]\n"
                f"• Test Duration:     {res.duration_seconds:.1f}s",
                title="Benchmark",
                border_style="green",
            )
        )
    else:
        console.print(f"[bold red]Benchmarking failed:[/bold red] {res.error_message}")
        raise typer.Exit(1)


# -------------------------------------------------------------------------
# INSPECTION COMMANDS
# -------------------------------------------------------------------------

@cli.command("inspect")
def inspect_cmd(
    model: Path = typer.Argument(..., help="Path to GGUF model"),
    show_tensors: bool = typer.Option(False, "--tensors", help="Display full tensor table"),
    breakdown: bool = typer.Option(False, "--breakdown", "-b", help="Display architectural layer memory breakdown"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Inspect metadata, architecture, and quantization details of a GGUF file."""
    reader = GGUFReader(model)
    load_t = show_tensors or breakdown
    info = reader.read_model_info(load_tensors=load_t)

    breakdown_data = reader.get_tensor_breakdown(info) if breakdown else None

    if json_output:
        data = {
            "file": str(info.file_path),
            "size_bytes": info.file_size_bytes,
            "architecture": info.architecture,
            "version": info.version,
            "context_length": info.context_length,
            "embedding_length": info.embedding_length,
            "tensor_count": info.tensor_count,
            "parameters": info.estimated_parameters,
            "dominant_quant": info.dominant_quant,
            "metadata_kv_count": info.metadata_kv_count,
            "breakdown": breakdown_data,
        }
        console.print(json.dumps(data, indent=2))
        return

    table = Table(title=f"GGUF Model Info: {model.name}", border_style="cyan")
    table.add_column("Property", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Architecture", info.architecture)
    table.add_row("Format Version", f"GGUF v{info.version}")
    table.add_row("Context Length", str(info.context_length or "N/A"))
    table.add_row("Embedding Length", str(info.embedding_length or "N/A"))
    table.add_row("Tensor Count", str(info.tensor_count))
    table.add_row("Dominant Quantization", info.dominant_quant or "Mixed")
    table.add_row("Estimated Parameters", f"{info.estimated_parameters / 1e9:.2f}B" if info.estimated_parameters else "N/A")
    table.add_row("File Size", f"{info.file_size_bytes / (1024**3):.2f} GB")

    console.print(table)

    if breakdown and breakdown_data:
        bd_table = Table(title="Layer Architectural Breakdown", border_style="emerald")
        bd_table.add_column("Layer Category", style="bold white")
        bd_table.add_column("Tensors", justify="right")
        bd_table.add_column("Parameters", justify="right")
        bd_table.add_column("% Total", justify="right")
        bd_table.add_column("Data Types", style="dim")

        for cat, d in breakdown_data.items():
            param_str = f"{d['params'] / 1e6:.1f}M" if d['params'] < 1e9 else f"{d['params'] / 1e9:.2f}B"
            types_str = ", ".join(d["types"][:3])
            bd_table.add_row(cat, str(d["count"]), param_str, f"{d['param_pct']}%", types_str)

        console.print(bd_table)


@cli.command("autotune")
def autotune_cmd(
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Optional model to calculate VRAM offload recommendation"),
):
    """Detect host CPU/GPU and display optimal threads and offload hyperparameters."""
    from quantforge.core.autotune import HardwareAutotuner
    from quantforge.formats.gguf_reader import GGUFReader

    tuner = HardwareAutotuner()
    prof = tuner.profile()

    model_size_gb = None
    layers = 32
    ctx = 4096
    if model and model.is_file():
        try:
            reader = GGUFReader(model)
            info = reader.read_model_info(load_tensors=False)
            model_size_gb = info.file_size_bytes / (1024**3)
            layers = info.block_count or 32
            ctx = info.context_length or 4096
        except Exception:
            model_size_gb = model.stat().st_size / (1024**3)

    rec = tuner.recommend(model_size_gb=model_size_gb, context_length=ctx, layer_count=layers)

    table = Table(title="QuantForge Hardware Auto-Tuner", border_style="cyan")
    table.add_column("Parameter", style="bold white")
    table.add_column("Detected / Recommended Value", style="cyan")

    table.add_row("Logical CPU Cores", str(prof.cpu_logical))
    table.add_row("Physical CPU Cores", str(prof.cpu_physical))
    table.add_row("Recommended CPU Threads (-t)", str(rec["optimal_threads"]))
    table.add_row("GPU Detected", "Yes" if prof.gpu_available else "None (CPU Backend)")
    if prof.gpu_available:
        table.add_row("GPU Device Name", prof.gpu_name or "N/A")
        table.add_row("Total GPU VRAM", f"{prof.gpu_vram_gb} GB")
        table.add_row("GPU Backend", prof.gpu_backend.upper())

    if model_size_gb is not None:
        table.add_row("Model Size", f"{model_size_gb:.2f} GB")
        table.add_row("Recommended GPU Layers (-ngl)", str(rec["recommended_n_gpu_layers"]))
        table.add_row("Fits 100% in VRAM", "[green]Yes[/green]" if rec["fits_fully_in_vram"] else "[yellow]Partial / CPU spill[/yellow]")

    console.print(table)


# -------------------------------------------------------------------------
# HUGGING FACE HUB COMMANDS
# -------------------------------------------------------------------------

@cli.command("pull")
def pull_cmd(
    repo_id: str = typer.Argument(..., help="Hugging Face repository (e.g. unsloth/Llama-3.2-3B-Instruct-GGUF)"),
    filename: Optional[str] = typer.Option(None, "--file", "-f", help="Specific GGUF file to download"),
    target_dir: Optional[Path] = typer.Option(None, "--dir", "-d", help="Destination folder"),
    token: Optional[str] = typer.Option(None, "--token", help="Hugging Face API token"),
):
    """Download a GGUF checkpoint directly from Hugging Face Hub."""
    from quantforge.formats.hf_hub import HFHubManager

    mgr = HFHubManager(token=token)
    console.print(f"[cyan]Downloading from Hugging Face Hub:[/cyan] [bold]{repo_id}[/bold]")
    with console.status("[bold green]Downloading checkpoint..."):
        try:
            local_path = mgr.pull_model(repo_id=repo_id, filename=filename, target_dir=target_dir)
            console.print(f"[bold green]Model saved to:[/bold green] {local_path}")
        except Exception as e:
            console.print(f"[bold red]Download failed:[/bold red] {e}")
            raise typer.Exit(1)


@cli.command("push")
def push_cmd(
    source_path: Path = typer.Argument(..., help="File or directory to upload"),
    repo: str = typer.Option(..., "--repo", "-r", help="Target Hugging Face repository (e.g. username/my-model-GGUF)"),
    token: Optional[str] = typer.Option(None, "--token", help="Hugging Face API token"),
    private: bool = typer.Option(False, "--private", help="Create private repository"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Commit message"),
):
    """Publish quantized models and Model Card to Hugging Face Hub."""
    from quantforge.formats.hf_hub import HFHubManager

    mgr = HFHubManager(token=token)
    console.print(f"[cyan]Uploading to Hugging Face Hub:[/cyan] [bold]{repo}[/bold]")
    with console.status("[bold green]Uploading files..."):
        try:
            url = mgr.push_models(repo_id=repo, source_path=source_path, commit_message=message, private=private)
            console.print(f"[bold green]Published successfully at:[/bold green] {url}")
        except Exception as e:
            console.print(f"[bold red]Publish failed:[/bold red] {e}")
            raise typer.Exit(1)


# -------------------------------------------------------------------------
# CALIBRATION DATASET COMMANDS
# -------------------------------------------------------------------------

@dataset_cli.command("list")
def dataset_list():
    """List built-in calibration dataset presets."""
    from quantforge.engines.dataset_manager import DatasetManager

    mgr = DatasetManager()
    presets = mgr.list_presets()

    table = Table(title="Calibration Dataset Presets", border_style="cyan")
    table.add_column("Preset Name", style="bold white")
    table.add_column("Category", style="cyan")
    table.add_column("Description", style="dim")

    for p in presets:
        table.add_row(p.name, p.category, p.description)

    console.print(table)


@dataset_cli.command("get")
def dataset_get(
    name: str = typer.Argument(..., help="Preset name (e.g. general-wiki, code-multilang)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Target output text path"),
    repeats: int = typer.Option(30, "--repeats", "-r", help="Number of repetitions for corpus size"),
):
    """Generate a calibration corpus file from a preset."""
    from quantforge.engines.dataset_manager import DatasetManager

    mgr = DatasetManager()
    try:
        out_path = mgr.generate_corpus(name, target_path=output, repeats=repeats)
        size_kb = out_path.stat().st_size / 1024
        console.print(f"[bold green]Corpus created successfully:[/bold green] {out_path} ({size_kb:.1f} KB)")
    except Exception as e:
        console.print(f"[bold red]Failed to generate corpus:[/bold red] {e}")
        raise typer.Exit(1)


@dataset_cli.command("prepare")
def dataset_prepare(
    files: List[Path] = typer.Argument(..., help="List of input text files to merge and clean"),
    output: Path = typer.Option(..., "--output", "-o", help="Destination merged corpus file"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable duplicate paragraph filtering"),
):
    """Merge and deduplicate raw text files into a single calibration corpus."""
    from quantforge.engines.dataset_manager import DatasetManager

    mgr = DatasetManager()
    out = mgr.prepare_corpus(files, output_path=output, deduplicate=not no_dedup)
    size_kb = out.stat().st_size / 1024
    console.print(f"[bold green]Cleaned corpus prepared:[/bold green] {out} ({size_kb:.1f} KB)")


# -------------------------------------------------------------------------
# PIPELINE COMMANDS
# -------------------------------------------------------------------------

@pipeline_cli.command("init")
def pipeline_init(
    template: str = typer.Option("balanced", "--template", "-t", help="Template: balanced, extreme"),
    output: Path = typer.Option(Path("./recipe.yaml"), "--output", "-o", help="Target recipe path"),
):
    """Generate a starter pipeline recipe YAML file."""
    recipe = EXTREME_COMPRESSION_RECIPE if template == "extreme" else BALANCED_RECIPE
    dump_recipe_to_yaml(recipe, output)
    console.print(f"[bold green]Recipe initialized at:[/bold green] {output}")


@pipeline_cli.command("run")
def pipeline_run(
    recipe_file: Path = typer.Argument(..., help="Path to recipe YAML file"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Override output directory"),
):
    """Execute an automated pipeline recipe."""
    runner = PipelineRunner()
    console.print(f"[bold cyan]Launching automated pipeline recipe:[/bold cyan] {recipe_file.name}")

    def _prog(p):
        console.print(f"[dim][{p.current_step}/{p.total_steps}][/dim] [bold]{p.stage.upper()}:[/bold] {p.detail}")

    summary = asyncio.run(runner.run_recipe_file(recipe_file, override_output_dir=output_dir, on_progress=_prog))

    if summary.success:
        console.print(
            Panel(
                f"[bold green]Pipeline Completed Successfully![/bold green]\n\n"
                f"• Generated Models: {len([q for q in summary.quant_results if q.success])}\n"
                f"• Model Card: [cyan]{summary.model_card_path}[/cyan]\n"
                f"• Output Directory: [cyan]{summary.output_dir}[/cyan]",
                title="Pipeline Success",
                border_style="green",
            )
        )
    else:
        console.print(f"[bold red]Pipeline failed:[/bold red] {summary.error_message}")
        raise typer.Exit(1)


# -------------------------------------------------------------------------
# WEB DASHBOARD COMMAND
# -------------------------------------------------------------------------

@cli.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable live code reload"),
):
    """Start the QuantForge web dashboard."""
    import uvicorn
    console.print(f"[bold green]Starting QuantForge Web Dashboard at http://{host}:{port}[/bold green]")
    uvicorn.run("quantforge.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()

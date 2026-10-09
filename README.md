# QuantForge

QuantForge is an automated GGUF quantization, imatrix calibration, and perplexity evaluation lab for local LLMs. It acts as an optimization companion to InferOps and llama.cpp, turning unquantized models or FP16 checkpoints into benchmarked quantization sets with measured quality-loss metrics.

## Key Capabilities

- **Quantization Engine**: Automates single and matrix runs across all modern llama.cpp quants: legacy formats (Q4_0, Q5_0, Q8_0), K-quants (Q2_K to Q6_K), and importance-matrix formats (IQ1_S through IQ4_NL). Supports mixed-precision per-tensor overrides (`--tensor-type`).
- **Imatrix Calibration Lab & Dataset Hub**: Generates custom `.dat` importance matrices using configurable calibration corpora, chunk sizes, and context windows. Includes built-in curated presets (`general-wiki`, `code-multilang`, `reasoning-math`, `multilingual-mixed`) and corpus deduplication tools.
- **Hugging Face Hub Integration**: Direct pull of unquantized models and automated publishing of complete quantization suites with generated model cards to Hugging Face Hub repositories.
- **Perplexity & Throughput Profiler**: Evaluates models against validation datasets, tracking delta perplexity relative to the FP16 baseline, and profiles token throughput with `llama-bench`.
- **Pareto Trade-off Analyzer & SVG Charts**: Computes the size versus perplexity versus speed frontier, generating standalone publication-ready vector SVG charts embedded directly into exported Model Cards.
- **Hardware Autotuner**: Benchmarks system CPU cores, RAM, and GPU VRAM to calculate optimal thread counts and layer offload allocation (`-ngl`) for quantization and evaluation jobs.
- **VRAM & KV Cache Simulator**: Calculates model weights, KV cache memory footprint (FP16, Q8_0, Q4_0), CUDA overhead, and GPU hardware compatibility matrix across context windows up to 128K.
- **Ollama & InferOps Exporter**: Generates production-ready `Modelfile` with inferred chat templates and stop tokens, plus InferOps runtime service specifications.
- **Canary Quality Benchmarks**: Runs automated zero-shot validation testing structured JSON output, Python code syntax, reasoning, and n-gram repetition degradation.
- **GGUF Sharding & Splitting**: Automates volume partitioning and reassembly for large 70B+ models exceeding single-file filesystem and repository limits.
- **LoRA Adapter Merge Engine**: Merges fine-tuned LoRA checkpoints directly into base unquantized GGUF checkpoints before quantization.
- **Standalone HTML Benchmark Reports**: Exports zero-dependency, single-file offline HTML optimization reports with embedded SVG charts and an interactive VRAM simulator.
- **Multi-Interface**: Rich Typer CLI with interactive tables and progress spinners, plus a FastAPI dashboard featuring a Visual Pipeline Stepper, Model Library, Studio, and Pareto frontier charts.

## Quick Start

### Installation

Clone the repository and install dependencies inside a virtual environment:

```bash
git clone https://github.com/alexandrmotologa/quantforge.git
cd quantforge

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

Verify your llama.cpp binary installation:

```bash
quantforge bin check
```

If binaries are missing, download official precompiled releases:

```bash
quantforge bin download --backend cpu
# or for CUDA
quantforge bin download --backend cuda
```

### Basic Quantization

Quantize an FP16 GGUF checkpoint to `Q4_K_M`:

```bash
quantforge quantize ./models/mistral-7b-fp16.gguf ./models/mistral-7b-q4km.gguf --quant Q4_K_M
```

Run a full quantization matrix:

```bash
quantforge matrix ./models/mistral-7b-fp16.gguf --quants Q4_K_M,Q5_K_M,IQ3_M,Q8_0 --output-dir ./dist
```

### Imatrix Calibration

Generate an importance matrix with a text corpus:

```bash
quantforge imatrix ./models/mistral-7b-fp16.gguf --data ./data/calibration.txt --output ./models/mistral-7b.dat --ctx-size 2048
```

Quantize using the calibrated matrix:

```bash
quantforge quantize ./models/mistral-7b-fp16.gguf ./models/mistral-7b-iq3m.gguf --quant IQ3_M --imatrix ./models/mistral-7b.dat
```

### Quality and Perplexity Evaluation

Calculate perplexity against an evaluation text file:

```bash
quantforge eval ./models/mistral-7b-q4km.gguf --data ./data/wikitext-2-raw/wiki.test.raw
```

Generate a comparative Pareto report across multiple quants:

```bash
quantforge pareto ./dist --base ./models/mistral-7b-fp16.gguf --output ./dist/pareto_report.json
```

### Hardware Autotuning

Detect host CPU cores, memory limits, and GPU VRAM to calculate optimal threading and layer offloading:

```bash
quantforge autotune --model ./models/mistral-7b-fp16.gguf
```

### Calibration Dataset Hub

List and download pre-curated imatrix calibration corpora, or merge domains:

```bash
quantforge dataset list
quantforge dataset download general-wiki --output-dir ./data
quantforge dataset merge ./data/code.txt ./data/wiki.txt --output ./data/combined.txt
```

### Hugging Face Hub Pull & Push

Download source checkpoints directly from Hugging Face Hub:

```bash
quantforge pull unsloth/mistral-7b-instruct-v0.3-GGUF --output-dir ./models
```

Publish a quantized suite with generated model cards and Pareto SVG:

```bash
quantforge push ./dist/mistral-7b my-org/mistral-7b-gguf-suite --token $HF_TOKEN
```

### Inspect Layer Breakdown

Examine tensor distributions and attention/FFN layer breakdowns:

```bash
quantforge inspect ./models/mistral-7b-q4km.gguf -b
```

### VRAM & KV Cache Simulator

Calculate precise weights, KV cache (FP16, Q8_0, Q4_0), CUDA overhead, and GPU tier compatibility:

```bash
quantforge vram-calc ./models/mistral-7b-q4km.gguf --ctx-size 8192 --kv-quant q8_0
```

### Export to Ollama & InferOps

Generate an Ollama `Modelfile` with extracted chat templates and stop tokens:

```bash
quantforge export ollama ./models/mistral-7b-q4km.gguf --output ./Modelfile
```

Generate an InferOps runtime service manifest:

```bash
quantforge export inferops ./models/mistral-7b-q4km.gguf --output ./inferops.yaml
```

### LoRA Merge Engine

Merge fine-tuned LoRA adapters directly into a base GGUF checkpoint:

```bash
quantforge lora merge --base ./models/base-fp16.gguf --lora ./adapters/math-lora.bin --scale 1.0 --output ./models/merged-fp16.gguf
```

### Canary & Quality Degeneration Benchmark

Run automated zero-shot battery prompts (JSON parsing, code syntax, logic riddle, repetition 4-grams) to detect quality loss:

```bash
quantforge canary ./models/mistral-7b-iq3m.gguf
```

### GGUF Sharding & Splitting

Partition large 70B+ checkpoints into volume shards for easier storage and upload:

```bash
quantforge split ./models/llama-70b-q4km.gguf --output-prefix ./dist/sharded/llama-70b --max-size 4G
```

Reassemble volume shards into a single unified model:

```bash
quantforge merge-shards ./dist/sharded/llama-70b-00001-of-00004.gguf --output ./models/llama-70b-merged.gguf
```

### Standalone HTML Benchmark Report

Export an offline, zero-dependency HTML optimization report with embedded SVG Pareto charts and an interactive VRAM simulator:

```bash
quantforge report ./dist/mistral-7b --output ./dist/mistral-7b/report.html
```

### Web Dashboard

Start the local web dashboard:

```bash
quantforge serve --port 8000
```

Open `http://localhost:8000` in your browser to inspect models, launch jobs, monitor live tensor progress, track visual pipelines at `/pipelines`, and explore the model library at `/models`.

## Automated Pipelines

QuantForge supports YAML recipes to run calibration, multi-quant generation, perplexity evaluation, and model card export in one pass:

```yaml
name: mistral-production-matrix
input_model: ./models/mistral-7b-fp16.gguf
calibration:
  dataset: ./data/calibration.txt
  chunks: 64
  context_size: 2048
quants:
  - Q4_K_M
  - Q5_K_M
  - IQ3_M
  - Q8_0
evaluation:
  dataset: ./data/eval.txt
  calc_pareto: true
output_dir: ./dist/mistral-7b
export_model_card: true
```

Execute the pipeline:

```bash
quantforge pipeline run ./recipes/production.yaml
```

## Architecture

QuantForge follows a modular architecture separating binary discovery, streaming execution, format analysis, and evaluation metrics:

```
src/quantforge/
├── core/
│   ├── config.py           # Configuration schemas and typed enums
│   ├── binary_manager.py   # Discovery, download, and verification of llama.cpp binaries
│   ├── executor.py         # Subprocess execution with real-time output parsing
│   ├── autotune.py         # CPU, memory, and GPU hardware profiling & offload heuristics
│   ├── vram_calc.py        # Weights, KV cache, and GPU hardware compatibility matrix
│   └── db.py               # SQLite storage for models, jobs, and benchmarks
├── engines/
│   ├── quantize.py         # llama-quantize execution and progress tracking
│   ├── imatrix.py          # llama-imatrix calibration pipeline
│   ├── dataset_manager.py  # Calibration dataset presets hub and deduplicated merger
│   ├── perplexity.py       # llama-perplexity evaluation and delta-PPL calculation
│   ├── benchmark.py        # llama-bench throughput profiling
│   ├── pareto.py           # Pareto frontier analysis and efficiency scoring
│   ├── canary.py           # Zero-shot validation prompts and degeneration testing
│   └── lora.py             # LoRA adapter merge engine
├── formats/
│   ├── gguf_reader.py      # Pure Python GGUF header parser and tensor breakdown analyzer
│   ├── pareto_svg.py       # Procedural standalone vector SVG Pareto chart generator
│   ├── card_generator.py   # Model card generator with embedded SVG & benchmark tables
│   ├── exporter.py         # Ollama Modelfile and InferOps runtime manifest generator
│   ├── splitter.py         # GGUF model volume sharding and reassembly
│   ├── html_report.py      # Standalone offline HTML benchmark report generator
│   └── hf_hub.py           # Hugging Face Hub download & suite publisher
├── pipelines/
│   ├── runner.py           # Multi-stage recipe orchestrator
│   └── templates.py        # Standard quantization recipes
├── cli/
│   └── app.py              # Typer CLI commands and Rich terminal tables
└── web/
    ├── app.py              # FastAPI application
    ├── api/                # REST endpoints and WebSocket progress feeds
    ├── static/             # Dashboard styles and scripts
    └── templates/          # Web views: Dashboard, Studio, Pipelines, Models, Pareto
```

## Supported Quantization Types

| Type | Bits | Typical Use Case | Recommended Imatrix |
|---|---|---|---|
| IQ1_S / IQ1_M | ~1.5 - 1.7 | Extreme edge constraints | Required |
| IQ2_XXS / IQ2_XS / IQ2_M | ~2.0 - 2.5 | Heavy memory limits | Required |
| IQ3_XXS / IQ3_S / IQ3_M | ~3.0 - 3.5 | Compact deployment with high retention | Strongly recommended |
| Q4_K_S / Q4_K_M | ~4.5 | Standard balanced inference | Recommended |
| IQ4_XS / IQ4_NL | ~4.2 - 4.5 | Enhanced 4-bit fidelity | Strongly recommended |
| Q5_K_S / Q5_K_M | ~5.5 | High-fidelity deployment | Optional |
| Q6_K | ~6.5 | Near-lossless general use | Optional |
| Q8_0 | 8.5 | Reference quant, negligible loss | Not required |

## Testing

Run the test suite with pytest:

```bash
pytest --cov=quantforge --cov-report=term-missing
```

## License

MIT License. See [LICENSE](LICENSE) for details.

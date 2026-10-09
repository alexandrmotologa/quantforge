# Architecture Specification

QuantForge orchestrates GGUF model conversion, importance matrix calibration, and quality evaluation. The system operates locally, interacting with native llama.cpp compiled executables while offering Python interfaces, CLI workflows, and a web management interface.

## System Overview

The application is structured into four main operational layers:

```
[ Typer CLI ]               [ FastAPI Web Dashboard ]
        \                           /
         \                         /
          v                       v
       [ Pipelines & Workflow Orchestrator ]
                     |
       [ Engine Layer (Quantize, Imatrix, Perplexity, Bench, Pareto) ]
                     |
  +------------------+------------------+
  |                                     |
  v                                     v
[ Binary Manager & Streaming Executor ]  [ SQLite Database & File Storage ]
  |
  v
[ llama.cpp Native Binaries (llama-quantize, llama-imatrix, llama-perplexity) ]
```

## Core Modules

### 1. Core Runtime (`quantforge.core`)

- **`config.py`**: Declares Pydantic V2 settings, enumerations (`QuantType`, `JobStatus`), and job configurations. Handles environment variable overrides and path defaults.
- **`binary_manager.py`**: Resolves local paths for `llama-quantize`, `llama-imatrix`, `llama-perplexity`, and `llama-bench`. When binaries are missing, it can download verified release archives from GitHub, unpack them into the local cache directory, and inspect their capabilities.
- **`executor.py`**: Subprocess runner with asynchronous streaming. Reads stdout and stderr line-by-line, runs regex patterns to capture real-time progress (tensor index, percentage, elapsed time, current perplexity estimate), and dispatches callbacks to listeners.
- **`db.py`**: SQLite database backed by SQLAlchemy. Stores models, quantization runs, calibration records, perplexity scores, and full execution logs for traceability.

### 2. Optimization Engines (`quantforge.engines`)

- **`quantize.py`**: Manages `llama-quantize` executions. Validates source GGUF files, applies optional importance matrices, handles extra flags like `--leave-output-tensor` or `--pure`, and emits structured metrics.
- **`imatrix.py`**: Manages `llama-imatrix` calibration. Prepares calibration corpora, splits text into token chunks, configures context lengths and GPU offloading, and verifies generated `.dat` matrix files.
- **`perplexity.py`**: Runs `llama-perplexity` against evaluation texts. Parses progressive chunk outputs, calculates mean perplexity with standard error, and compares against baseline FP16 metrics.
- **`benchmark.py`**: Wraps `llama-bench` to capture token generation speed across prompt processing and text generation phases.
- **`pareto.py`**: Evaluates model trade-offs across size, perplexity degradation, and inference speed. Identifies Pareto-optimal quants and computes a Quality-per-Gigabyte efficiency score.

### 3. Format & Export Layer (`quantforge.formats`)

- **`gguf_reader.py`**: Pure Python GGUF header parser. Reads metadata keys, tensor counts, architecture type, context length, and alignment without requiring external tools.
- **`card_generator.py`**: Formats evaluation results and quant specs into standard Markdown model cards suitable for Hugging Face Hub repositories.

### 4. Pipelines & Automation (`quantforge.pipelines`)

- **`runner.py`**: Reads YAML pipeline manifests and executes multi-stage workflows: validation, imatrix calibration, matrix quantization, perplexity measurement, and model card export.
- **`templates.py`**: Provides preconfigured workflows for standard balance, low-memory edge devices, and full-spectrum distribution suites.

### 5. Interfaces (`quantforge.cli` and `quantforge.web`)

- **CLI (`quantforge.cli`)**: Built with Typer and Rich. Supports interactive tables, progress bars, and exit codes for CI/CD integration.
- **Web UI (`quantforge.web`)**: FastAPI application serving HTML views and JSON endpoints. Includes real-time job log streaming via WebSockets and SSE.

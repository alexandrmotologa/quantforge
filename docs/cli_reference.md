# CLI Reference

QuantForge provides the `quantforge` command line tool powered by Typer and Rich.

## Commands Overview

```bash
quantforge [OPTIONS] COMMAND [ARGS]...
```

### Options

- `--version`: Show QuantForge version.
- `--help`: Show help message.

---

## `quantforge quantize`

Quantizes a single GGUF model into a specified format.

```bash
quantforge quantize <INPUT_MODEL> <OUTPUT_MODEL> [OPTIONS]
```

### Arguments

- `INPUT_MODEL`: Path to source unquantized or FP16 GGUF model.
- `OUTPUT_MODEL`: Destination path for the quantized GGUF file.

### Options

- `-q, --quant <TYPE>`: Target quantization format (e.g., `Q4_K_M`, `Q5_K_M`, `IQ3_M`). Default is `Q4_K_M`.
- `-i, --imatrix <FILE>`: Path to `.dat` importance matrix file for calibration-aware quantization.
- `--leave-output-tensor`: Leave output tensor in FP32/FP16 (recommended for embeddings/logits). Default is false.
- `--pure`: Force quantization of all tensors, including normalization layers. Default is false.
- `-t, --threads <N>`: Number of CPU threads to use. Default is auto-detected.

---

## `quantforge matrix`

Executes a batch quantization run, creating multiple quant variants from a single base model.

```bash
quantforge matrix <INPUT_MODEL> [OPTIONS]
```

### Options

- `--quants <LIST>`: Comma-separated list of target formats (e.g. `Q4_K_M,Q5_K_M,IQ3_M,Q8_0`).
- `-o, --output-dir <DIR>`: Output directory for quantized models. Default is `./quants`.
- `-i, --imatrix <FILE>`: Optional importance matrix applied to relevant formats.
- `-t, --threads <N>`: Thread count.

---

## `quantforge imatrix`

Calibrates an importance matrix using a reference text corpus.

```bash
quantforge imatrix <INPUT_MODEL> [OPTIONS]
```

### Options

- `-d, --data <FILE>`: Path to plain text or jsonl calibration dataset.
- `-o, --output <FILE>`: Destination path for `.dat` importance matrix file.
- `-c, --ctx-size <N>`: Context window size for calibration passes. Default is 2048.
- `--chunks <N>`: Number of text chunks to process. Default is 64.
- `-ngl, --n-gpu-layers <N>`: GPU offload layers. Default is 0.
- `-t, --threads <N>`: Thread count.

---

## `quantforge eval`

Calculates perplexity on an evaluation corpus.

```bash
quantforge eval <MODEL_PATH> [OPTIONS]
```

### Options

- `-d, --data <FILE>`: Path to test dataset (e.g. wikitext raw test split).
- `-c, --ctx-size <N>`: Evaluation context size. Default is 2048.
- `-ngl, --n-gpu-layers <N>`: Number of layers to offload to GPU. Default is 0.
- `-b, --batch-size <N>`: Batch size for token evaluation. Default is 512.
- `--base <BASE_MODEL>`: Optional baseline model path to calculate delta perplexity.

---

## `quantforge bench`

Profiles inference speed using `llama-bench`.

```bash
quantforge bench <MODEL_PATH> [OPTIONS]
```

### Options

- `-p, --prompt-tokens <N>`: Prompt length for evaluation. Default is 512.
- `-n, --gen-tokens <N>`: Generation tokens count. Default is 128.
- `-ngl, --n-gpu-layers <N>`: Number of GPU offloaded layers.

---

## `quantforge inspect`

Inspects GGUF file metadata, tensor list, and quantization details.

```bash
quantforge inspect <MODEL_PATH> [OPTIONS]
```

### Options

- `--show-tensors`: Print full tensor architecture list. Default is false.
- `--json`: Output metadata as structured JSON. Default is false.

---

## `quantforge pipeline run`

Executes a multi-stage automation recipe defined in YAML.

```bash
quantforge pipeline run <RECIPE_PATH> [OPTIONS]
```

### Options

- `--output-dir <DIR>`: Override output directory in recipe.
- `--skip-eval`: Skip perplexity evaluation stage.

---

## `quantforge bin`

Manages llama.cpp native binary discovery, inspection, and automatic installation.

```bash
quantforge bin check      # Verify all required binaries are found and runnable
quantforge bin download   # Download official precompiled releases from GitHub
quantforge bin path       # Print configured binary search paths
```

---

## `quantforge serve`

Starts the FastAPI web dashboard.

```bash
quantforge serve [OPTIONS]
```

### Options

- `--host <IP>`: Host interface to bind. Default is `127.0.0.1`.
- `-p, --port <PORT>`: Port number. Default is 8000.
- `--reload`: Enable auto-reload for development.

# REST API and WebSocket Reference

QuantForge serves a FastAPI application providing REST endpoints for model management, job dispatch, and live progress feeds.

## Base URL

By default, the server listens at `http://127.0.0.1:8000`.

Interactive OpenAPI documentation is available at `/docs` and `/redoc`.

## Endpoints

### 1. System and Binaries

#### `GET /api/system/status`
Returns system resources, CPU cores, GPU status (if available), and binary locator status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "binaries": {
    "llama_quantize": { "found": true, "path": "C:/tools/llama-quantize.exe", "version": "b3800" },
    "llama_imatrix": { "found": true, "path": "C:/tools/llama-imatrix.exe", "version": "b3800" },
    "llama_perplexity": { "found": true, "path": "C:/tools/llama-perplexity.exe", "version": "b3800" },
    "llama_bench": { "found": true, "path": "C:/tools/llama-bench.exe", "version": "b3800" }
  },
  "gpu_available": true,
  "cpu_count": 16
}
```

#### `POST /api/system/binaries/download`
Triggers background download and extraction of precompiled llama.cpp release binaries.

---

### 2. Models and GGUF Inspection

#### `GET /api/models`
Lists all registered GGUF models in configured search paths.

#### `GET /api/models/{model_id}`
Returns metadata, architecture, tensor counts, context window, and quantization details for a specific model.

---

### 3. Jobs and Quantization

#### `POST /api/jobs/quantize`
Dispatches a single or matrix quantization job.

**Request Body:**
```json
{
  "input_path": "/path/to/base-fp16.gguf",
  "output_path": "/path/to/output-q4km.gguf",
  "quant_type": "Q4_K_M",
  "imatrix_path": null,
  "leave_output_tensor": false,
  "pure": false,
  "threads": 8
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "job-8f2a41bc",
  "status": "queued",
  "created_at": "2026-10-09T18:00:00Z"
}
```

#### `POST /api/jobs/imatrix`
Dispatches an imatrix calibration job.

#### `GET /api/jobs`
Lists historical and active jobs with statuses and timestamps.

#### `GET /api/jobs/{job_id}`
Retrieves details, progress, and logs for a job.

#### `POST /api/jobs/{job_id}/cancel`
Signals cancellation to a running subprocess.

---

### 4. Evaluation and Pareto

#### `POST /api/eval/perplexity`
Dispatches a perplexity evaluation task.

#### `GET /api/eval/pareto`
Calculates Pareto efficiency metrics for a set of quantized models against a baseline.

---

### 5. WebSocket Live Stream

#### `WS /ws/jobs/{job_id}`
Subscribes to real-time events for an active job.

**Message format:**
```json
{
  "job_id": "job-8f2a41bc",
  "type": "progress",
  "progress_pct": 42.5,
  "current_tensor": "blk.12.attn_q.weight",
  "total_tensors": 291,
  "current_tensor_idx": 124,
  "raw_line": "[124/291] - blk.12.attn_q.weight: Q4_K_M"
}
```

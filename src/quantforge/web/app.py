"""FastAPI Web Application and REST API for QuantForge."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from quantforge import __version__
from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import (
    ImatrixConfig,
    JobStatus,
    JobType,
    PerplexityConfig,
    QuantizationConfig,
    QuantType,
    settings,
)
from quantforge.core.db import db, JobRecord
from quantforge.engines.imatrix import ImatrixEngine
from quantforge.engines.pareto import ParetoAnalyzer, QuantPoint
from quantforge.engines.perplexity import PerplexityEngine
from quantforge.engines.quantize import QuantizeEngine
from quantforge.formats.gguf_reader import GGUFReader

app = FastAPI(
    title="QuantForge Lab",
    description="Automated GGUF quantization, imatrix calibration, and perplexity evaluation lab.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

templates_dir.mkdir(parents=True, exist_ok=True)
static_dir.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# In-memory WebSocket connection manager for live job feeds
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)

    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

ws_manager = ConnectionManager()
binary_manager = BinaryManager()
quant_engine = QuantizeEngine(binary_manager)
imatrix_engine = ImatrixEngine(binary_manager)
ppl_engine = PerplexityEngine(binary_manager)
pareto_analyzer = ParetoAnalyzer()


# -------------------------------------------------------------------------
# REQUEST SCHEMAS
# -------------------------------------------------------------------------

class QuantizeRequest(BaseModel):
    input_path: str
    output_path: str
    quant_type: str = "Q4_K_M"
    imatrix_path: Optional[str] = None
    leave_output_tensor: bool = False
    pure: bool = False
    threads: Optional[int] = None


class ImatrixRequest(BaseModel):
    input_path: str
    data_path: str
    output_path: str
    ctx_size: int = 2048
    chunks: int = 64
    n_gpu_layers: int = 0
    threads: Optional[int] = None


class EvalRequest(BaseModel):
    model_path: str
    data_path: str
    ctx_size: int = 2048
    n_gpu_layers: int = 0
    baseline_ppl: Optional[float] = None


class InspectRequest(BaseModel):
    model_path: str


class ParetoRequest(BaseModel):
    points: List[Dict[str, Any]]
    baseline_quant: Optional[str] = None


# -------------------------------------------------------------------------
# REST API ENDPOINTS
# -------------------------------------------------------------------------

@app.get("/api/system/status")
async def get_system_status():
    """Inspects native binary status and system capabilities."""
    status = binary_manager.get_status()
    bin_dict = {}
    for k, v in status.items():
        bin_dict[k] = {
            "found": v.found,
            "path": str(v.path) if v.path else None,
            "version": v.version,
            "error": v.error,
        }

    return {
        "status": "healthy",
        "version": __version__,
        "cpu_count": os.cpu_count() or 4,
        "default_threads": settings.default_threads,
        "binaries": bin_dict,
    }


@app.post("/api/system/binaries/download")
async def trigger_binary_download(backend: str = "cpu", background_tasks: BackgroundTasks = None):
    """Triggers background download of official precompiled binaries."""
    def _run_download():
        binary_manager.download_release_binaries(backend=backend)

    if background_tasks:
        background_tasks.add_task(_run_download)
        return {"status": "started", "message": f"Downloading binaries for backend '{backend}' in background."}
    else:
        _run_download()
        return {"status": "completed"}


@app.get("/api/jobs")
async def list_jobs(limit: int = 50):
    """Lists recent quantization, calibration, and evaluation jobs."""
    jobs = db.list_jobs(limit=limit)
    return [
        {
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status,
            "input_model": j.input_model,
            "output_model": j.output_model,
            "quant_type": j.quant_type,
            "progress_pct": j.progress_pct,
            "current_step": j.current_step,
            "return_code": j.return_code,
            "error_message": j.error_message,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        }
        for j in jobs
    ]


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Retrieves full job details and recent logs."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    metrics = json.loads(job.metrics_json) if job.metrics_json else {}
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "input_model": job.input_model,
        "output_model": job.output_model,
        "quant_type": job.quant_type,
        "progress_pct": job.progress_pct,
        "current_step": job.current_step,
        "return_code": job.return_code,
        "logs": job.logs,
        "metrics": metrics,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@app.post("/api/jobs/quantize")
async def dispatch_quantize(req: QuantizeRequest, background_tasks: BackgroundTasks):
    """Dispatches a single quantization task."""
    try:
        q_type = QuantType(req.quant_type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid quantization type: {req.quant_type}")

    cfg = QuantizationConfig(
        input_path=Path(req.input_path),
        output_path=Path(req.output_path),
        quant_type=q_type,
        imatrix_path=Path(req.imatrix_path) if req.imatrix_path else None,
        leave_output_tensor=req.leave_output_tensor,
        pure=req.pure,
        threads=req.threads,
    )

    job_id = f"job-q-{uuid.uuid4().hex[:8]}"

    async def _runner():
        async def _prog(p):
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "percent": p.percent,
                "current_item": p.current_item,
                "raw_line": p.raw_line,
            })
        await quant_engine.quantize(cfg, job_id=job_id, on_progress=_prog)

    background_tasks.add_task(_runner)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/jobs/imatrix")
async def dispatch_imatrix(req: ImatrixRequest, background_tasks: BackgroundTasks):
    """Dispatches an importance matrix calibration task."""
    cfg = ImatrixConfig(
        input_path=Path(req.input_path),
        data_path=Path(req.data_path),
        output_path=Path(req.output_path),
        ctx_size=req.ctx_size,
        chunks=req.chunks,
        n_gpu_layers=req.n_gpu_layers,
        threads=req.threads,
    )

    job_id = f"job-im-{uuid.uuid4().hex[:8]}"

    async def _runner():
        async def _prog(p):
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "percent": p.percent,
                "current_item": p.current_item,
                "raw_line": p.raw_line,
            })
        await imatrix_engine.calibrate(cfg, job_id=job_id, on_progress=_prog)

    background_tasks.add_task(_runner)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/models/inspect")
async def inspect_model(req: InspectRequest):
    """Inspects metadata of a local GGUF file."""
    path = Path(req.model_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="GGUF model file not found")

    try:
        reader = GGUFReader(path)
        info = reader.read_model_info(load_tensors=False)
        return {
            "file": str(info.file_path),
            "size_bytes": info.file_size_bytes,
            "architecture": info.architecture,
            "version": info.version,
            "context_length": info.context_length,
            "embedding_length": info.embedding_length,
            "tensor_count": info.tensor_count,
            "parameters": info.estimated_parameters,
            "dominant_quant": info.dominant_quant,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse GGUF: {e}")


@app.post("/api/eval/pareto")
async def calculate_pareto(req: ParetoRequest):
    """Computes Pareto frontier for given quant evaluation points."""
    pts = []
    for item in req.points:
        pts.append(
            QuantPoint(
                quant_type=item.get("quant_type", "UNKNOWN"),
                file_path=Path(item.get("file_path", "")),
                file_size_gb=float(item.get("file_size_gb", 0.0)),
                perplexity=float(item.get("perplexity", 0.0)),
                speed_tok_s=float(item.get("speed_tok_s", 0.0)) if item.get("speed_tok_s") else None,
                is_baseline=bool(item.get("is_baseline", False)),
            )
        )
    report = pareto_analyzer.compute_frontier(pts, baseline_quant=req.baseline_quant)
    return report.to_dict()


# -------------------------------------------------------------------------
# WEBSOCKET PROGRESS STREAM
# -------------------------------------------------------------------------

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            # Keepalive listener
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)


# -------------------------------------------------------------------------
# WEB DASHBOARD VIEWS
# -------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def view_dashboard(request: Request):
    """Renders the main QuantForge dashboard."""
    jobs = db.list_jobs(limit=10)
    status = binary_manager.get_status()
    all_found = all(s.found for s in status.values())
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "jobs": jobs,
            "binaries": status,
            "all_binaries_found": all_found,
            "version": __version__,
        },
    )


@app.get("/studio", response_class=HTMLResponse)
async def view_studio(request: Request):
    """Renders the Quantization Studio interface."""
    return templates.TemplateResponse(
        request=request,
        name="studio.html",
        context={
            "quant_types": [q.value for q in QuantType],
            "version": __version__,
        },
    )


@app.get("/eval", response_class=HTMLResponse)
async def view_eval(request: Request):
    """Renders the Perplexity and Pareto frontier evaluation interface."""
    return templates.TemplateResponse(
        request=request,
        name="eval.html",
        context={
            "version": __version__,
        },
    )

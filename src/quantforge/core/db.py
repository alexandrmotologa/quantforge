"""Database schema, models, and session management for QuantForge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from quantforge.core.config import settings

Base = declarative_base()


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ModelRecord(Base):
    """Registered GGUF model checkpoint."""

    __tablename__ = "models"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False, unique=True)
    architecture = Column(String(64), nullable=True)
    quant_type = Column(String(32), nullable=True)
    size_bytes = Column(Integer, nullable=False)
    tensor_count = Column(Integer, nullable=True)
    context_length = Column(Integer, nullable=True)
    meta_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class JobRecord(Base):
    """Asynchronous quantization, imatrix, or evaluation job."""

    __tablename__ = "jobs"

    id = Column(String(64), primary_key=True)
    job_type = Column(String(32), nullable=False)
    status = Column(String(32), default="queued", nullable=False)
    input_model = Column(Text, nullable=True)
    output_model = Column(Text, nullable=True)
    quant_type = Column(String(32), nullable=True)
    progress_pct = Column(Float, default=0.0)
    current_step = Column(String(255), nullable=True)
    return_code = Column(Integer, nullable=True)
    logs = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class ImatrixRecord(Base):
    """Calibrated importance matrix file."""

    __tablename__ = "imatrices"

    id = Column(String(64), primary_key=True)
    model_path = Column(Text, nullable=False)
    dataset_path = Column(Text, nullable=False)
    output_path = Column(Text, nullable=False)
    ctx_size = Column(Integer, default=2048)
    chunks = Column(Integer, default=64)
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class EvalRecord(Base):
    """Quality and throughput evaluation benchmark result."""

    __tablename__ = "evaluations"

    id = Column(String(64), primary_key=True)
    model_path = Column(Text, nullable=False)
    quant_type = Column(String(32), nullable=False)
    dataset_name = Column(String(128), nullable=True)
    perplexity = Column(Float, nullable=True)
    perplexity_stderr = Column(Float, nullable=True)
    delta_perplexity = Column(Float, nullable=True)
    speed_prompt_tok_s = Column(Float, nullable=True)
    speed_gen_tok_s = Column(Float, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class DatabaseManager:
    """Manages SQLite connection lifecycle and queries."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = (db_path or settings.database_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.init_db()

    def init_db(self) -> None:
        """Creates tables if they do not exist."""
        Base.metadata.create_all(self.engine)

    def create_job(
        self,
        job_id: str,
        job_type: str,
        input_model: Optional[str] = None,
        output_model: Optional[str] = None,
        quant_type: Optional[str] = None,
    ) -> JobRecord:
        """Registers a new job record."""
        with self.SessionFactory() as session:
            record = JobRecord(
                id=job_id,
                job_type=job_type,
                status="queued",
                input_model=input_model,
                output_model=output_model,
                quant_type=quant_type,
                created_at=utc_now(),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def update_job_progress(
        self,
        job_id: str,
        progress_pct: float,
        current_step: Optional[str] = None,
        log_line: Optional[str] = None,
    ) -> None:
        """Updates real-time progress for a running job."""
        with self.SessionFactory() as session:
            job = session.get(JobRecord, job_id)
            if job:
                job.progress_pct = round(progress_pct, 2)
                if current_step:
                    job.current_step = current_step
                if log_line:
                    existing = job.logs or ""
                    job.logs = (existing + "\n" + log_line)[-50000:]
                session.commit()

    def finalize_job(
        self,
        job_id: str,
        status: str,
        return_code: int,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        full_logs: Optional[str] = None,
    ) -> None:
        """Marks a job as finished."""
        with self.SessionFactory() as session:
            job = session.get(JobRecord, job_id)
            if job:
                job.status = status
                job.return_code = return_code
                job.error_message = error_message
                job.finished_at = utc_now()
                if metrics:
                    job.metrics_json = json.dumps(metrics)
                if full_logs:
                    job.logs = full_logs[-100000:]
                session.commit()

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        """Returns recent jobs."""
        with self.SessionFactory() as session:
            stmt = select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)
            return list(session.scalars(stmt).all())

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Gets a job by ID."""
        with self.SessionFactory() as session:
            return session.get(JobRecord, job_id)


# Global database manager instance
db = DatabaseManager()

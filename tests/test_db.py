"""Tests for DatabaseManager operations and models."""

from pathlib import Path
from quantforge.core.db import DatabaseManager, JobRecord, utc_now


def test_database_crud(tmp_path: Path):
    db_file = tmp_path / "test.sqlite"
    db_mgr = DatabaseManager(db_path=db_file)

    # 1. Create job
    job = db_mgr.create_job(
        job_id="job-101",
        job_type="quantize",
        input_model="/path/base.gguf",
        output_model="/path/q4km.gguf",
        quant_type="Q4_K_M",
    )
    assert job.id == "job-101"
    assert job.status == "queued"

    # 2. Update progress
    db_mgr.update_job_progress(
        job_id="job-101",
        progress_pct=50.5,
        current_step="blk.12.attn_q.weight",
        log_line="[12/24] quantizing tensor",
    )
    fetched = db_mgr.get_job("job-101")
    assert fetched is not None
    assert fetched.progress_pct == 50.5
    assert fetched.current_step == "blk.12.attn_q.weight"
    assert "[12/24]" in fetched.logs

    # 3. Finalize job
    db_mgr.finalize_job(
        job_id="job-101",
        status="completed",
        return_code=0,
        metrics={"ratio": 3.4},
        full_logs="full execution log",
    )
    finished = db_mgr.get_job("job-101")
    assert finished.status == "completed"
    assert finished.return_code == 0
    assert "ratio" in finished.metrics_json

    # 4. List jobs
    all_jobs = db_mgr.list_jobs(limit=10)
    assert len(all_jobs) >= 1
    assert any(j.id == "job-101" for j in all_jobs)

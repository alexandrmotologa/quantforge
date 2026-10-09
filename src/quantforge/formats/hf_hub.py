"""Hugging Face Hub integration for pulling checkpoints and publishing quant suites."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import logging

from huggingface_hub import HfApi, hf_hub_download

from quantforge.core.config import settings

logger = logging.getLogger(__name__)


class HFHubManager:
    """Manages downloading base models and uploading quantized packages to Hugging Face Hub."""

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.api = HfApi(token=self.token)

    def list_gguf_files(self, repo_id: str) -> List[str]:
        """Lists all .gguf files inside a Hugging Face repository."""
        files = self.api.list_repo_files(repo_id=repo_id, token=self.token)
        return [f for f in files if f.lower().endswith(".gguf")]

    def pull_model(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        target_dir: Optional[Path] = None,
    ) -> Path:
        """Downloads a GGUF file from Hugging Face Hub."""
        dest_dir = (target_dir or settings.data_dir / "models").resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        selected_file = filename
        if not selected_file:
            candidates = self.list_gguf_files(repo_id)
            if not candidates:
                raise FileNotFoundError(f"No .gguf files found in Hugging Face repository '{repo_id}'")
            # Prefer unquantized / f16 / bf16 if present
            preferred = [c for c in candidates if any(t in c.lower() for t in ["f16", "fp16", "bf16", "f32"])]
            selected_file = preferred[0] if preferred else candidates[0]

        logger.info(f"Downloading {selected_file} from {repo_id}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=selected_file,
            local_dir=str(dest_dir),
            token=self.token,
        )
        return Path(downloaded_path).resolve()

    def push_models(
        self,
        repo_id: str,
        source_path: Path,
        commit_message: Optional[str] = None,
        private: bool = False,
    ) -> str:
        """Publishes quantized GGUF models and model card documentation to Hugging Face Hub."""
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source path not found: {src}")

        # Ensure repo exists
        self.api.create_repo(
            repo_id=repo_id,
            token=self.token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )

        msg = commit_message or "Upload quantized GGUF suite via QuantForge"

        if src.is_file():
            self.api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=src.name,
                repo_id=repo_id,
                token=self.token,
                commit_message=msg,
            )
        elif src.is_dir():
            # Upload folder contents (all .gguf and README.md)
            self.api.upload_folder(
                folder_path=str(src),
                repo_id=repo_id,
                token=self.token,
                commit_message=msg,
                allow_patterns=["*.gguf", "*.md", "*.svg", "*.json", "*.txt"],
            )

        return f"https://huggingface.co/{repo_id}"

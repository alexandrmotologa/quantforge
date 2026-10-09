"""Binary discovery, validation, and automated acquisition for llama.cpp executables."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import zipfile
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import logging

import httpx

from quantforge.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BinaryInfo:
    """Status details for a resolved executable."""

    name: str
    found: bool
    path: Optional[Path] = None
    version: Optional[str] = None
    is_executable: bool = False
    error: Optional[str] = None


class BinaryManager:
    """Locates, verifies, and downloads native llama.cpp executables."""

    BINARY_NAMES = {
        "quantize": ["llama-quantize", "quantize"],
        "imatrix": ["llama-imatrix", "imatrix"],
        "perplexity": ["llama-perplexity", "perplexity"],
        "bench": ["llama-bench"],
        "cli": ["llama-cli", "main"],
    }

    def __init__(self, custom_dir: Optional[Path] = None) -> None:
        self.custom_dir = custom_dir or settings.binaries_path

    def _executable_names(self, base_name: str) -> List[str]:
        """Returns candidate filenames taking platform extensions into account."""
        is_win = platform.system().lower() == "windows"
        exts = [".exe"] if is_win else [""]
        variants = []
        for name in self.BINARY_NAMES.get(base_name, [base_name]):
            variants.append(name)
            for ext in exts:
                if not name.endswith(ext):
                    variants.append(f"{name}{ext}")
        return list(dict.fromkeys(variants))

    def get_search_paths(self) -> List[Path]:
        """Returns ordered list of directories to inspect for binaries."""
        paths: List[Path] = []

        # 1. Custom or managed directory
        if self.custom_dir:
            paths.append(self.custom_dir)

        # 2. Environment variables
        for env_var in ("LLAMA_CPP_DIR", "LLAMA_BIN_DIR", "LLAMACPP_PATH"):
            val = os.environ.get(env_var)
            if val:
                paths.append(Path(val).resolve())

        # 3. System PATH directories
        system_path = os.environ.get("PATH", "")
        for item in system_path.split(os.pathsep):
            if item.strip():
                paths.append(Path(item.strip()).resolve())

        # Deduplicate while preserving order
        unique_paths: List[Path] = []
        seen = set()
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        return unique_paths

    def find_binary(self, binary_key: str) -> Optional[Path]:
        """Finds path for a specific binary key (e.g. 'quantize', 'imatrix')."""
        candidates = self._executable_names(binary_key)
        for search_dir in self.get_search_paths():
            if not search_dir.is_dir():
                continue
            for cand in candidates:
                target = search_dir / cand
                if target.is_file() and os.access(target, os.X_OK | os.R_OK):
                    return target.resolve()
        return None

    def probe_version(self, binary_path: Path) -> Optional[str]:
        """Runs the binary with --version or --help to parse its build tag."""
        for flag in ["--version", "-h", "--help"]:
            try:
                res = subprocess.run(
                    [str(binary_path), flag],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    check=False,
                )
                output = (res.stdout + "\n" + res.stderr).strip()
                for line in output.splitlines():
                    lower_line = line.lower()
                    if "version:" in lower_line or "build:" in lower_line or "commit:" in lower_line:
                        return line.strip()
                    if "llama.cpp" in lower_line and ("b" in lower_line or "version" in lower_line):
                        return line.strip()
            except Exception:
                continue
        return "available (build info unspecified)"

    def inspect_binary(self, binary_key: str) -> BinaryInfo:
        """Inspects status and version for a given tool key."""
        path = self.find_binary(binary_key)
        if not path:
            return BinaryInfo(
                name=binary_key,
                found=False,
                error=f"Executable '{binary_key}' not found in search paths.",
            )

        version = self.probe_version(path)
        return BinaryInfo(
            name=binary_key,
            found=True,
            path=path,
            version=version,
            is_executable=True,
        )

    def get_status(self) -> Dict[str, BinaryInfo]:
        """Inspects all required and optional llama.cpp executables."""
        results: Dict[str, BinaryInfo] = {}
        for key in self.BINARY_NAMES.keys():
            results[key] = self.inspect_binary(key)
        return results

    def require_binary(self, binary_key: str) -> Path:
        """Returns path or raises a clear FileNotFoundError."""
        path = self.find_binary(binary_key)
        if not path:
            search_str = "\n".join(f"  - {p}" for p in self.get_search_paths()[:6])
            raise FileNotFoundError(
                f"Required native tool '{binary_key}' was not found.\n"
                f"Run 'quantforge bin download' to download precompiled binaries, or set LLAMA_CPP_DIR.\n"
                f"Searched locations:\n{search_str}"
            )
        return path

    def download_release_binaries(
        self,
        backend: str = "cpu",
        target_dir: Optional[Path] = None,
        tag: str = "latest",
    ) -> Path:
        """Downloads official precompiled release from GitHub and unpacks to target_dir."""
        dest_dir = (target_dir or self.custom_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        system = platform.system().lower()
        machine = platform.machine().lower()

        # Resolve asset pattern
        if system == "windows":
            if backend == "cuda":
                asset_pattern = "bin-win-cuda"
            elif backend == "vulkan":
                asset_pattern = "bin-win-vulkan"
            else:
                asset_pattern = "bin-win-avx2-x64.zip" if "64" in machine else "bin-win"
        elif system == "linux":
            asset_pattern = "bin-ubuntu-x64.zip" if "64" in machine else "bin-ubuntu"
        elif system == "darwin":
            asset_pattern = "bin-macos-arm64.zip" if "arm" in machine else "bin-macos-x64.zip"
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")

        # Fetch releases list to find the latest with valid assets
        repo_urls = [
            "https://api.github.com/repos/ggml-org/llama.cpp/releases",
            "https://api.github.com/repos/ggerganov/llama.cpp/releases",
        ]

        release_data = None
        matched_asset = None
        headers = {"User-Agent": "QuantForge-BinaryManager/1.0"}

        with httpx.Client(headers=headers, follow_redirects=True, timeout=20.0) as client:
            for repo_url in repo_urls:
                try:
                    url = f"{repo_url}?per_page=5"
                    resp = client.get(url)
                    if resp.status_code == 200:
                        releases_list = resp.json()
                        for rel in releases_list:
                            assets = rel.get("assets", [])
                            # Search for matching asset in this release
                            for asset in assets:
                                a_name = asset["name"].lower()
                                # Skip auxiliary cudart dll archives
                                if a_name.startswith("cudart"):
                                    continue
                                if "arm64" in a_name and "arm" not in machine:
                                    continue
                                if ("x64" in a_name or "amd64" in a_name) and "arm" in machine:
                                    continue

                                if system == "windows":
                                    if backend == "cuda" and "bin-win-cuda" in a_name and "x64" in a_name:
                                        matched_asset = asset
                                        release_data = rel
                                        break
                                    elif backend == "vulkan" and "bin-win-vulkan" in a_name and "x64" in a_name:
                                        matched_asset = asset
                                        release_data = rel
                                        break
                                    elif backend == "cpu" and ("bin-win-cpu-x64" in a_name or "bin-win-avx2-x64" in a_name):
                                        matched_asset = asset
                                        release_data = rel
                                        break
                                    elif "bin-win" in a_name and "x64" in a_name and a_name.endswith(".zip"):
                                        matched_asset = asset
                                        release_data = rel
                                        break
                                elif system == "linux":
                                    if "bin-ubuntu-x64" in a_name or "bin-linux-x64" in a_name:
                                        matched_asset = asset
                                        release_data = rel
                                        break
                                elif system == "darwin":
                                    if "bin-macos" in a_name:
                                        matched_asset = asset
                                        release_data = rel
                                        break
                            if matched_asset:
                                break
                    if matched_asset:
                        break
                except Exception as e:
                    logger.warning(f"Failed to query {repo_url}: {e}")

        if not matched_asset:
            raise RuntimeError(
                f"Could not find matching precompiled release asset for {system}-{machine} ({backend}) in recent releases."
            )

        download_url = matched_asset["browser_download_url"]
        archive_name = matched_asset["name"]
        temp_archive_path = dest_dir / archive_name

        logger.info(f"Downloading {archive_name} from {download_url}...")
        with httpx.Client(headers=headers, follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", download_url) as response:
                response.raise_for_status()
                with open(temp_archive_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)

        # Unpack
        logger.info(f"Extracting {temp_archive_path} to {dest_dir}...")
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(temp_archive_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        elif archive_name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(temp_archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(dest_dir)
        else:
            raise RuntimeError(f"Unknown archive extension: {archive_name}")

        temp_archive_path.unlink(missing_ok=True)

        # If files were extracted into a nested directory, flatten executables
        for root, _, files in os.walk(dest_dir):
            for file in files:
                source_file = Path(root) / file
                if source_file.parent != dest_dir:
                    dest_file = dest_dir / file
                    if not dest_file.exists():
                        shutil.copy2(source_file, dest_file)

        return dest_dir

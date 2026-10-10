"""Collect a restricted baseline manifest without credentials or host identifiers.

Run in the authorized Linux compute environment. Does not install packages,
import accelerator frameworks, alter repositories, or run inference.
"""
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_value(path, *args):
    if not (path / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True,
            text=True, timeout=15, check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def collect():
    packages = {}
    for name in ("torch", "vllm", "vllm-plugin-fl", "flag-gems", "evalscope", "triton", "flagtree"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    repos = {}
    for name in ("vllm-plugin-FL", "FlagGems"):
        path = Path("/workspace") / name
        status = git_value(path, "status", "--porcelain")
        repos[name] = {
            "exists": path.is_dir(),
            "commit": git_value(path, "rev-parse", "HEAD"),
            "branch": git_value(path, "branch", "--show-current"),
            "dirty": bool(status) if status is not None else None,
        }
    model = Path("/workspace/MiniCPM5-2B")
    benchmark = Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_status": "not_run",
        "python": platform.python_version(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "packages": packages,
        "repos": repos,
        "model_exists": model.is_dir(),
        "model_config_sha256": digest(model / "config.json"),
        "model_tokenizer_config_sha256": digest(model / "tokenizer_config.json"),
        "dataset_exists": Path("/workspace/evalscope-datasets/math_500").is_dir(),
        "benchmark_sha256": digest(benchmark),
        "gpu_hardware_and_driver": "requires_separate_check_in_gpu_mode",
    }


if __name__ == "__main__":
    print(json.dumps(collect(), ensure_ascii=False, indent=2))

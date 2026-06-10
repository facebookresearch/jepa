"""Minimal reproducibility metadata for experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256_file(path: Path) -> str:
    """Return a SHA-256 digest, reusing a sidecar generated for large files."""
    sidecar = path.with_name(f"{path.name}.sha256")
    if sidecar.is_file():
        value = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if len(value) == 64:
            return value
    value = sha256_file(path)
    sidecar.write_text(f"{value}  {path.name}\n", encoding="utf-8")
    return value


def git_commit(repo_root: Path) -> str:
    """Return the current Git commit, including a dirty-worktree marker."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{commit}-dirty" if dirty else commit


def package_versions(names: list[str]) -> dict[str, str]:
    """Return installed versions for the requested distributions."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def base_manifest(repo_root: Path) -> dict[str, object]:
    """Build the shared, compact runtime manifest."""
    return {
        "git_commit": git_commit(repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(
            [
                "torch",
                "torchvision",
                "transformers",
                "numpy",
                "pandas",
                "scikit-learn",
                "opencv-python",
            ]
        ),
    }


def save_manifest(data: dict[str, object], path: Path) -> None:
    """Persist a readable JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

"""Shared fixtures.

The parity tests compare the binding against the `litsea` CLI, which is the
reference implementation for what a model should produce. The CLI is built
once per session so the comparison always runs rather than being skipped.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """Return the directory holding the bundled models."""
    assert MODELS_DIR.is_dir(), f"missing models directory: {MODELS_DIR}"
    return MODELS_DIR


@pytest.fixture(scope="session")
def litsea_cli() -> Path:
    """Build the `litsea` CLI and return the path to the binary."""
    subprocess.run(
        ["cargo", "build", "--quiet", "-p", "litsea-cli"],
        cwd=REPO_ROOT,
        check=True,
    )
    binary = REPO_ROOT / "target" / "debug" / "litsea"
    assert binary.is_file(), f"cargo build did not produce {binary}"
    return binary


def run_cli(binary: Path, args: list[str], stdin: str) -> list[str]:
    """Run the CLI over `stdin` and return its output lines.

    Args:
        binary: Path to the `litsea` binary.
        args: Arguments after the subcommand name.
        stdin: Text to feed on standard input.

    Returns:
        The output lines, with the trailing newline removed.
    """
    result = subprocess.run(
        [str(binary), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()

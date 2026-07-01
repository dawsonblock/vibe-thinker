"""Wheel-install smoke test — verifies the installed package is operational.

Builds the wheel from the current source tree, installs it into a fresh
virtual environment, and exercises the installed entry points:

  - vibe-thinker --help
  - vibe-thinker doctor
  - vibe-thinker smoke
  - vibe-thinker finalize-migration --help
  - vibe-thinker-ui --help
  - web UI static file is reachable from the installed package

This test is intentionally integration-level and slow. It is marked
``integration`` so it does not run in the fast core/local gates.
"""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_wheel(dist_dir: Path) -> Path:
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheel found in {dist_dir}")
    return wheels[0]


def _build_wheel(tmp_dir: Path) -> Path:
    root = _project_root()
    dist_dir = tmp_dir / "dist"
    try:
        import build
    except ImportError as e:
        pytest.skip(f"`build` package not installed: {e}")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return _find_wheel(dist_dir)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_bin(venv_dir: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _run_in_venv(venv_dir: Path, cmd: list, check: bool = True, **kwargs):
    python = _venv_python(venv_dir)
    return subprocess.run(
        [str(python), "-m"] + cmd,
        check=check,
        capture_output=True,
        text=True,
        **kwargs,
    )


def _run_python(venv_dir: Path, cmd: list, check: bool = True, **kwargs):
    python = _venv_python(venv_dir)
    return subprocess.run(
        [str(python)] + cmd,
        check=check,
        capture_output=True,
        text=True,
        **kwargs,
    )


def _run_cmd(venv_dir: Path, cmd: list, check: bool = True, **kwargs):
    return subprocess.run(
        [str(_venv_bin(venv_dir, cmd[0]))] + cmd[1:],
        check=check,
        capture_output=True,
        text=True,
        **kwargs,
    )


def test_wheel_install_cli_and_web():
    root = _project_root()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wheel = _build_wheel(tmp_dir)

        venv_dir = tmp_dir / "venv"
        venv.create(venv_dir, with_pip=True)

        # Upgrade pip/setuptools/wheel first.
        _run_in_venv(
            venv_dir,
            ["pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"],
        )
        # Install the wheel (no extras — core only).
        _run_in_venv(venv_dir, ["pip", "install", "--quiet", str(wheel)])

        # --- CLI smoke ---
        r = _run_cmd(venv_dir, ["vibe-thinker", "--help"])
        assert "usage:" in r.stdout.lower()

        r = _run_cmd(venv_dir, ["vibe-thinker", "doctor"])
        assert "core local profile is runnable" in r.stdout

        r = _run_cmd(venv_dir, ["vibe-thinker", "smoke"])
        assert "smoke test passed" in r.stdout.lower()

        # --- finalize-migration help / module loadable ---
        r = _run_cmd(
            venv_dir,
            ["vibe-thinker", "finalize-migration", "--help"],
        )
        assert "recall" in r.stdout.lower()

        # --- web UI entry point ---
        r = _run_cmd(venv_dir, ["vibe-thinker-ui", "--help"])
        assert "vibe-thinker" in r.stdout.lower()

        # --- web/static/index.html is present in the installed package ---
        r = _run_python(
            venv_dir,
            ["-c",
             "from pathlib import Path; "
             "import web; "
             "p = Path(web.__file__).parent / 'static' / 'index.html'; "
             "assert p.exists(), f'missing {p}'; "
             "print('static_ok', p.stat().st_size)"],
        )
        assert "static_ok" in r.stdout

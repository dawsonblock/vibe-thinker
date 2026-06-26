"""Sandbox execution layer for CodeVerifier.

This package provides isolated code execution for verifying untrusted
model output. The architecture is defense-in-depth:

    ┌─────────────────────────────────────────┐
    │  Docker Sandbox (sbx microVM)           │  ← outer layer
    │  ┌───────────────────────────────────┐  │
    │  │  vibe-thinker orchestrator        │  │
    │  │  ┌─────────────────────────────┐  │  │
    │  │  │  CodeVerifier               │  │  │
    │  │  │  ┌───────────────────────┐  │  │  │
    │  │  │  │  DockerSandboxExecutor│  │  │  │  ← inner layer
    │  │  │  │  docker run --network=none │  │
    │  │  │  │  --memory=128m --read-only │  │
    │  │  │  │  --security-opt=no-new-... │  │
    │  │  │  └───────────────────────┘  │  │
    │  │  └─────────────────────────────┘  │
    │  └───────────────────────────────────┘
    └─────────────────────────────────────────┘

Executors:
  - DockerSandboxExecutor: runs Python in a Docker container with
    network isolation, memory limits, read-only filesystem, and no
    new privileges. This is the production executor.
  - LocalSubprocessExecutor: runs Python in a local subprocess with
    timeout only. This is NOT safe for untrusted code — it exists for
    development and testing where Docker is not available.

Fallback policy: if no sandbox executor is available, CodeVerifier
REFUSES to verify — it does not fall back to host execution. Running
untrusted model output directly on the host is unacceptable.
"""

from sandbox.base import ExecutionResult, SandboxExecutor
from sandbox.docker_executor import DockerSandboxExecutor
from sandbox.local_executor import LocalSubprocessExecutor
from sandbox.sbx_executor import DockerSbxExecutor
from sandbox.warm_pool_executor import WarmDockerPool

__all__ = [
    "ExecutionResult",
    "SandboxExecutor",
    "DockerSandboxExecutor",
    "DockerSbxExecutor",
    "LocalSubprocessExecutor",
    "WarmDockerPool",
]

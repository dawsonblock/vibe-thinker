"""Deterministic code verifier.

Executes Python snippets in a bounded subprocess and captures stdout/stderr.
Can run provided unit tests against a candidate solution. Fails closed on
timeout or execution error.

Security: code is run in a subprocess with a timeout. This is NOT a full
sandbox — for untrusted code, use a proper sandbox (Docker, nsjail, etc.).
The subprocess approach prevents the verifier from hanging the main process
on infinite loops, but does not isolate filesystem or network access.

Capabilities:
  - run small Python snippets in subprocess
  - timeout execution (default 5 seconds)
  - capture stdout/stderr
  - run provided unit tests
  - fail closed on timeout/error
"""

import asyncio
import textwrap
from typing import Any, Dict, Optional

from verifiers.base import VerificationResult


class CodeVerifier:
    """Deterministic verifier that executes Python code in a subprocess."""

    name = "code_verifier"

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def verify(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> VerificationResult:
        unit_tests = context.get("unit_tests")
        expected_output = context.get("expected_output")

        # If unit tests are provided, run them against the candidate code.
        if unit_tests:
            return await self._run_unit_tests(answer, unit_tests)

        # If expected_output is provided, run the code and compare stdout.
        if expected_output is not None:
            return await self._run_and_compare_stdout(answer, str(expected_output))

        # No verification criteria provided — be honest.
        return VerificationResult(
            verified=False,
            score=0.0,
            method="python_exec",
            evidence={"candidate_code": answer[:200]},
            error="no unit_tests or expected_output provided; cannot verify",
        )

    async def _run_unit_tests(
        self, code: str, unit_tests: str
    ) -> VerificationResult:
        """Run unit tests against the candidate code."""
        # Dedent the candidate code and tests, then indent them under
        # the try blocks. We strip leading/trailing blank lines first to
        # avoid indentation errors from stray newlines.
        code_clean = textwrap.dedent(code).strip()
        tests_clean = textwrap.dedent(unit_tests).strip()
        script = (
            "import sys, traceback\n"
            "try:\n"
            + textwrap.indent(code_clean, "    ") + "\n"
            "except Exception as e:\n"
            "    print(f'IMPORT_ERROR: {e}')\n"
            "    sys.exit(1)\n"
            "try:\n"
            + textwrap.indent(tests_clean, "    ") + "\n"
            "    print('ALL_TESTS_PASSED')\n"
            "except AssertionError as e:\n"
            "    print(f'ASSERTION_FAILED: {e}')\n"
            "    sys.exit(1)\n"
            "except Exception as e:\n"
            "    print(f'TEST_ERROR: {e}')\n"
            "    traceback.print_exc()\n"
            "    sys.exit(1)\n"
        )
        return await self._execute(script, method="unit_tests",
                                    success_marker="ALL_TESTS_PASSED",
                                    evidence={"code_length": len(code),
                                              "tests_length": len(unit_tests)})

    async def _run_and_compare_stdout(
        self, code: str, expected_output: str
    ) -> VerificationResult:
        """Run code and compare stdout to expected output."""
        script = code
        result = await self._execute_raw(script)
        if result.error:
            return result
        actual = result.evidence.get("stdout", "").strip()
        expected = expected_output.strip()
        if actual == expected:
            return VerificationResult(
                verified=True,
                score=1.0,
                method="python_exec",
                evidence={"stdout": actual, "expected": expected},
            )
        return VerificationResult(
            verified=False,
            score=0.0,
            method="python_exec",
            evidence={"stdout": actual, "expected": expected},
            error=f"stdout mismatch: got {actual!r}, expected {expected!r}",
        )

    async def _execute(
        self, script: str, method: str, success_marker: str,
        evidence: Optional[Dict] = None
    ) -> VerificationResult:
        """Execute a script and check for a success marker in stdout.

        Does NOT bail early on non-zero exit codes — assertion failures
        exit with code 1 but still print useful messages to stdout.
        """
        raw = await self._execute_raw(script)
        # Timeout or execution failure (no stdout at all) -> bail
        if raw.error and "timed out" in raw.error:
            return raw
        if raw.error and "execution failed" in raw.error:
            return raw
        stdout = raw.evidence.get("stdout", "")
        stderr = raw.evidence.get("stderr", "")
        returncode = raw.evidence.get("returncode", 0)
        if success_marker in stdout:
            return VerificationResult(
                verified=True,
                score=1.0,
                method=method,
                evidence={**(evidence or {}), "stdout": stdout,
                          "returncode": returncode},
            )
        # Non-zero exit without success marker -> include the error detail
        error_msg = f"success marker {success_marker!r} not found in stdout"
        if returncode != 0:
            # Include assertion/test error from stdout if present
            for marker in ("ASSERTION_FAILED", "IMPORT_ERROR", "TEST_ERROR"):
                if marker in stdout:
                    error_msg = stdout.strip()
                    break
            else:
                error_msg = f"process exited with code {returncode}: {error_msg}"
        return VerificationResult(
            verified=False,
            score=0.0,
            method=method,
            evidence={**(evidence or {}), "stdout": stdout,
                      "stderr": stderr, "returncode": returncode},
            error=error_msg,
        )

    async def _execute_raw(self, script: str) -> VerificationResult:
        """Execute a Python script in a subprocess with timeout.

        Returns a result with stdout/stderr in evidence. The ``verified``
        field is always False here — callers check for success markers.
        Non-zero exit codes are noted in the error but stdout is still
        returned so callers can check for assertion failure messages.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return VerificationResult(
                    verified=False,
                    score=0.0,
                    method="python_exec",
                    evidence={"timeout": self.timeout},
                    error=f"execution timed out after {self.timeout}s",
                )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            returncode = proc.returncode
            # Return stdout/stderr regardless of exit code — callers need
            # to check for success markers or assertion failure messages.
            error = None
            if returncode != 0:
                error = f"process exited with code {returncode}"
            return VerificationResult(
                verified=False,  # caller checks for success marker
                score=0.0,
                method="python_exec",
                evidence={"stdout": stdout, "stderr": stderr,
                          "returncode": returncode},
                error=error,
            )
        except Exception as e:
            return VerificationResult(
                verified=False,
                score=0.0,
                method="python_exec",
                evidence={},
                error=f"execution failed: {e}",
            )

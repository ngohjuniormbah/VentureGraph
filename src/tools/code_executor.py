"""
Python execution tool for the TAM Calculator.

The TAM Calculator agent (`src.agents.venture_catalyst.calculate_tam`) needs
to turn numbers it finds via web search (market size, growth rate, customer
counts) into an actual TAM figure. Rather than asking Claude to "do the
math" in its head - where arithmetic mistakes and unit errors are common and
invisible - it is given this `execute_python` tool: it writes a short script
that computes the figure, we actually run it, and the real, verifiable
stdout is what gets reported as the TAM. `TAMEstimate.calculation_code` and
`TAMEstimate.calculation_output` (see `src/schemas/venture.py`) preserve both
sides of that, so the arithmetic is auditable rather than asserted.

Security note: this runs model-generated code in a separate OS process
(not `exec()` in the calling process) with `python3 -I` (isolated mode,
which ignores `PYTHONPATH`/user site-packages) and a wall-clock timeout, but
it is **not** a full security sandbox - the subprocess still has the same
filesystem/network access as the host process. That's an acceptable trust
boundary for a single-user local tool computing arithmetic, matching how
Anthropic's own and OpenAI's code-execution tools work. For a shared,
multi-tenant deployment, this should instead run inside a locked-down
container/VM (e.g. gVisor, Firecracker, or a network-isolated Docker
container) rather than a bare subprocess.
"""

import asyncio
import sys


async def execute_python(code: str, timeout: float = 10.0) -> str:
    """
    Execute a Python script in an isolated subprocess and return its output.

    Data flow:
        1. Receives `code` - a short script Claude wrote inside the TAM
           Calculator's agentic tool loop (see
           `src.agents.tool_loop.run_agentic_tool_loop`).
        2. Spawns `python3 -I -c <code>` as a child process via
           `asyncio.create_subprocess_exec` (no shell involved, so there's
           no shell-injection surface from the code string itself).
        3. Waits for it to finish, bounded by `timeout` seconds; kills it
           and returns an error string if it runs too long.
        4. Returns the script's captured stdout (what the calling agent
           should `print()` its result to) - or a formatted error message
           if the script raised an exception or timed out.

    Args:
        code: A self-contained Python script. Must `print()` whatever value
            should be reported back to the calling agent.
        timeout: Maximum wall-clock seconds to let the script run before
            it's killed.

    Returns:
        The script's stdout on success, or a string describing the error
        (non-zero exit, timeout, or failure to launch) on failure. Errors
        are returned as strings rather than raised, so the calling agent
        (in the tool loop) sees them as a tool result and can react (e.g.
        by fixing and re-running the script) instead of the whole pipeline
        crashing.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return f"Error: failed to start Python subprocess: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Error: execution exceeded {timeout}s timeout and was killed."

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        return f"Error (exit code {proc.returncode}):\n{stderr_text or stdout_text}"

    return stdout_text or "(script ran successfully but printed nothing - use print() to return a value)"

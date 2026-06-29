"""Inter-turn tool-callback scaffold for the reasoning loop (v3.2 Step 4.2).

This implements the practical version of "multi-turn specialist/generalist
dialog": INTER-TURN tool calls (generate -> tool call -> continue), NOT
mid-token pausing. Mid-token pausing is genuinely hard with llama-cpp
streaming (you'd have to suspend generation, inject tool output, resume
the KV cache) and the win over inter-turn is marginal. Inter-turn is the
right boundary: the specialist generates a full turn, if it emitted a
tool-call marker we execute the tool, append the result, and re-prompt.

Design:
  - A ``ToolCallback`` protocol: ``async def __call__(query: str) -> str``.
    Tools register under a name (e.g. "generalist_query", "rag_lookup").
  - A ``CallbackRegistry`` maps tool names to callables.
  - The specialist signals a tool call by emitting a marker line:
        <tool_call name="generalist_query">some fact I need</tool_call>
    The parser extracts (name, query) from this marker.
  - ``run_with_callbacks`` drives the loop: call the generator, scan for
    tool-call markers, execute each, append the results to the
    conversation as a system note, re-prompt. Bounded by ``max_rounds``
    (default 3) to prevent infinite tool-call loops. After the rounds are
    exhausted OR no marker is found, the final generation is returned.

This is a SCAFFOLD: the orchestrator wires the registry (e.g. mapping
"generalist_query" to a FactualVerifier RAG call) and passes it to the
specialist invocation. Without a registry, the loop is a no-op and the
specialist's marker is left in the output untouched (backward-compat).

Fail-safe:
  - An unknown tool name -> the call is skipped with an error note, the
    loop continues (the specialist sees the error and can recover).
  - A tool raising an exception -> the exception text is fed back as the
    tool result so the specialist can adapt, NOT propagated to crash the
    trajectory.
  - max_rounds is a hard ceiling — a specialist stuck in a tool-call
    loop is cut off and its last generation returned as-is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Protocol

# The marker the specialist emits to request a tool call. Chosen to be
# unlikely to appear in natural reasoning text and easy to regex out.
# Format: <tool_call name="tool_name">the query/argument</tool_call>
_TOOL_CALL_RE = re.compile(
    r'<tool_call\s+name="(?P<name>[a-zA-Z0-9_\-]+)\s*"\s*>'
    r'(?P<query>.*?)</tool_call>',
    re.DOTALL,
)


class ToolCallback(Protocol):
    """A tool the specialist can invoke between generation turns.

    The callable takes the tool-query string and returns the tool's
    result string (which is fed back into the conversation). It MUST
    be async — tools may do network I/O (RAG fetch, generalist query).
    """

    async def __call__(self, query: str) -> str: ...


@dataclass
class CallbackRegistry:
    """Maps tool names to ToolCallback callables.

    Empty by default -> the callback loop is a no-op (backward-compat).
    The orchestrator populates this with real tools (e.g.
    "generalist_query" -> a FactualVerifier RAG call).
    """

    tools: Dict[str, Callable[[str], Awaitable[str]]] = field(default_factory=dict)

    def register(self, name: str, callback: Callable[[str], Awaitable[str]]) -> None:
        self.tools[name] = callback

    def get(self, name: str) -> Optional[Callable[[str], Awaitable[str]]]:
        return self.tools.get(name)


@dataclass
class ToolCallRequest:
    """A parsed tool-call request from the specialist's output."""

    name: str
    query: str
    start: int  # char offset in the source text
    end: int    # char offset (exclusive)


def parse_tool_calls(text: str) -> List[ToolCallRequest]:
    """Find all <tool_call name="...">...</tool_call> markers in text.

    Returns them in order of appearance. Used by run_with_callbacks to
    decide whether a generation turn requested any tools.
    """
    calls: List[ToolCallRequest] = []
    for m in _TOOL_CALL_RE.finditer(text):
        calls.append(ToolCallRequest(
            name=m.group("name").strip(),
            query=m.group("query").strip(),
            start=m.start(),
            end=m.end(),
        ))
    return calls


def strip_tool_call_markers(text: str) -> str:
    """Remove all tool-call markers from text.

    Used for the FINAL output (after the loop) so the markers don't
    leak into the answer the user/verifier sees.
    """
    return _TOOL_CALL_RE.sub("", text)


async def execute_tool_call(
    registry: CallbackRegistry, request: ToolCallRequest,
) -> str:
    """Execute one tool call, fail-safe.

    Unknown tool -> an error note. Tool raises -> the exception text.
    Never propagates: the specialist sees the error and can recover.
    """
    callback = registry.get(request.name)
    if callback is None:
        return f"[tool error: unknown tool '{request.name}']"
    try:
        return await callback(request.query)
    except Exception as e:
        return f"[tool error: {type(e).__name__}: {e}]"


async def run_with_callbacks(
    generate: Callable[[str], Awaitable[str]],
    initial_prompt: str,
    registry: Optional[CallbackRegistry] = None,
    max_rounds: int = 3,
) -> str:
    """Drive the generate -> tool-call -> continue loop.

    Args:
        generate: async callable that takes a prompt and returns the
            model's generation. Called once per round.
        initial_prompt: the first prompt to generate from.
        registry: the tool registry. If None or empty, the loop runs
            exactly once (no callbacks) — backward-compat with the
            pre-v3.2 single-turn specialist call.
        max_rounds: hard ceiling on generate rounds. Default 3 prevents
            infinite tool-call loops. After exhaustion, the last
            generation is returned with markers stripped.

    Returns:
        The final generation with tool-call markers stripped. Tool
        results are visible in the intermediate turns (fed back to the
        model) but not in the returned string.

    The conversation is built by appending, after each generation that
    contained tool calls, a `<tool_result name="...">...</tool_result>`
    block for each call, then re-prompting the model to continue. This
    is the inter-turn boundary — the model generates a full turn, we
    handle the tools, then it generates again.
    """
    if registry is None or not registry.tools:
        # No callbacks configured -> single-turn, backward-compat.
        return await generate(initial_prompt)

    prompt = initial_prompt
    last_generation = ""
    for _ in range(max_rounds):
        generation = await generate(prompt)
        last_generation = generation
        calls = parse_tool_calls(generation)
        if not calls:
            # No tool calls requested -> done.
            return generation
        # Execute each tool call and append the results as a continuation
        # of the conversation. The model sees its own generation (with
        # the markers) plus the tool results, then continues.
        tool_results = []
        for call in calls:
            result = await execute_tool_call(registry, call)
            tool_results.append(
                f'<tool_result name="{call.name}">{result}</tool_result>'
            )
        # Build the next prompt: original + this generation + tool results
        # + a continue instruction.
        prompt = (
            prompt
            + generation
            + "\n" + "\n".join(tool_results) + "\n"
            + "Continue your reasoning using these tool results.\n"
        )
    # Exhausted rounds — return the last generation, markers stripped.
    return strip_tool_call_markers(last_generation)

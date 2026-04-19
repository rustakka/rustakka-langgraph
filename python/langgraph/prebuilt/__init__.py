"""``langgraph.prebuilt`` — Python helpers built on the native engine.

These wrappers replicate the ergonomics of the upstream prebuilt package
while the graph topology and execution still run in Rust.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from rustakka_langgraph import END, START
from rustakka_langgraph.graph_builder import build_react_agent_graph


class ToolNode:
    """Wraps a list of tools so the graph can invoke them in parallel.

    A "tool" is any callable (or object exposing ``name`` + ``invoke``) that
    accepts a dict of arguments and returns a JSON-serialisable value.
    """

    def __init__(self, tools: Iterable[Any]):
        self._tools: dict[str, Callable[[dict], Any]] = {}
        for t in tools:
            name = getattr(t, "name", None) or t.__name__
            invoke = getattr(t, "invoke", None) or t
            self._tools[name] = invoke

    def __call__(self, state: dict) -> dict:
        messages = state.get("messages", []) or []
        last = messages[-1] if messages else {}
        calls = last.get("tool_calls", []) if isinstance(last, dict) else []
        out = []
        for c in calls:
            name = c.get("name")
            args = c.get("args", {}) or {}
            fn = self._tools.get(name)
            if fn is None:
                content = f"error: unknown tool `{name}`"
            else:
                try:
                    content = fn(args)
                except Exception as exc:  # pragma: no cover
                    content = f"error: {exc}"
            out.append({"id": c.get("id"), "type": "tool", "name": name, "content": content})
        return {"messages": out}


def tools_condition(state: dict) -> str:
    """Route to ``tools`` if the last assistant message has ``tool_calls``."""
    messages = state.get("messages", []) or []
    last = messages[-1] if messages else {}
    if isinstance(last, dict) and last.get("tool_calls"):
        return "tools"
    return END


def create_react_agent(
    model: Callable[[list, Any], dict],
    tools: Iterable[Any],
    *,
    system_prompt: str | None = None,
    recursion_limit: int | None = None,
) -> Any:
    """Assemble a ReAct-style ``agent → tools → agent`` loop.

    ``model(messages, system_prompt) -> assistant_message`` is the contract.
    Tools follow the ``ToolNode`` contract.
    """

    return build_react_agent_graph(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        recursion_limit=recursion_limit,
    )


__all__ = ["ToolNode", "tools_condition", "create_react_agent"]

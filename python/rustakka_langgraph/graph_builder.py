"""Pure-Python helpers that wire user callables into a native StateGraph."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from . import _native as _n


def build_react_agent_graph(
    *,
    model: Callable[[list, Any], dict],
    tools: Iterable[Any],
    system_prompt: str | None = None,
    recursion_limit: int | None = None,
) -> Any:
    """Construct the canonical ReAct graph using the native engine.

    The topology matches ``rustakka_langgraph_prebuilt::create_react_agent``:
    a single ``agent`` node calls the user-supplied model, emits an assistant
    message, and routes through ``tools_condition`` to either the tool
    executor or ``END``.
    """
    from langgraph.prebuilt import ToolNode, tools_condition

    g = _n.StateGraph()

    def agent(state: dict) -> dict:
        messages = list(state.get("messages", []) or [])
        assistant = model(messages, system_prompt)
        return {"messages": [assistant]}

    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))

    g.add_edge(_n.START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    g.add_edge("agent", _n.END)
    return g.compile()


__all__ = ["build_react_agent_graph"]

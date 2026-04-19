"""End-to-end Python tests for the langgraph facade."""

from __future__ import annotations

import pytest

from langgraph import END, START, MemorySaver
from langgraph.graph import StateGraph


def test_invoke_two_node_graph() -> None:
    g = StateGraph()
    g.add_node("a", lambda state: {"x": 1})
    g.add_node("b", lambda state: {"x": state.get("x", 0) + 41})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile()
    out = app.invoke({})
    assert out["x"] == 42


def test_checkpointer_resumes_state_across_invocations() -> None:
    g = StateGraph()
    g.add_node("inc", lambda state: {"count": state.get("count", 0) + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    saver = MemorySaver()
    app = g.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t1"}}
    r1 = app.invoke({}, config=cfg)
    r2 = app.invoke({}, config=cfg)
    assert r1["count"] == 1
    assert r2["count"] == 2


def test_stream_emits_values() -> None:
    g = StateGraph()
    g.add_node("a", lambda state: {"x": 1})
    g.add_edge(START, "a")
    g.add_edge("a", END)
    app = g.compile()
    events = list(app.stream({}, stream_mode="values"))
    assert any(ev.get("kind") == "values" for ev in events)


def test_compile_requires_nodes() -> None:
    g = StateGraph()
    g.add_edge(START, END)
    with pytest.raises(Exception):
        g.compile()

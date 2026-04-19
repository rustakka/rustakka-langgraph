"""Python tests for prebuilt helpers and the functional API."""

from __future__ import annotations

from langgraph import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition


def test_tools_condition_routes_to_tools_when_calls_present() -> None:
    state = {"messages": [{"tool_calls": [{"id": "1", "name": "x", "args": {}}]}]}
    assert tools_condition(state) == "tools"


def test_tools_condition_routes_to_end_when_idle() -> None:
    state = {"messages": [{"role": "assistant", "content": "done"}]}
    assert tools_condition(state) != "tools"


def test_tool_node_executes_calls() -> None:
    def add(args: dict) -> int:
        return args["a"] + args["b"]

    add.name = "add"  # type: ignore[attr-defined]
    node = ToolNode([add])
    out = node(
        {
            "messages": [
                {
                    "tool_calls": [
                        {"id": "c1", "name": "add", "args": {"a": 2, "b": 40}},
                    ]
                }
            ]
        }
    )
    assert out["messages"][0]["content"] == 42


def test_react_agent_two_turn_loop() -> None:
    turns = {"n": 0}

    def model(messages, system_prompt):
        turns["n"] += 1
        if turns["n"] == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "add", "args": {"a": 40, "b": 2}}],
            }
        return {"role": "assistant", "content": "the answer is 42"}

    def add(args: dict) -> int:
        return args["a"] + args["b"]

    add.name = "add"  # type: ignore[attr-defined]

    app = create_react_agent(model, [add], recursion_limit=10)
    out = app.invoke({"messages": [{"role": "user", "content": "40+2?"}]})
    last = out["messages"][-1]
    assert last["content"] == "the answer is 42"
    assert turns["n"] == 2


def test_entrypoint_decorates_a_callable_graph() -> None:
    @entrypoint()
    def flow(state: dict) -> dict:
        return {"x": state.get("x", 0) + 1}

    assert flow({"x": 1})["x"] == 2


def test_entrypoint_with_checkpointer_resumes_state() -> None:
    saver = MemorySaver()

    @entrypoint(checkpointer=saver)
    def flow(state: dict) -> dict:
        return {"count": state.get("count", 0) + 1}

    cfg = {"configurable": {"thread_id": "tfunc"}}
    assert flow.invoke({}, config=cfg)["count"] == 1
    assert flow.invoke({}, config=cfg)["count"] == 2


def test_task_decorator_preserves_callable() -> None:
    @task
    def square(x: int) -> int:
        return x * x

    assert square(5) == 25
    assert getattr(square, "__rustakka_langgraph_task__", False) is True

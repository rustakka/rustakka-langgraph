"""``langgraph.func`` — functional API (``@entrypoint``, ``@task``).

This mirrors upstream's functional decorator layer on top of the native
engine. An ``@entrypoint`` compiles a single-node graph whose body is the
decorated function; ``@task`` marks helpers invoked from inside that body.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from rustakka_langgraph import END, START
from rustakka_langgraph import _native as _n


def task(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``func`` as a subtask of an ``@entrypoint``.

    Subtasks execute inline under the same graph step; the decorator merely
    tags the callable so future versions can schedule them as fan-out
    targets.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.__rustakka_langgraph_task__ = True  # type: ignore[attr-defined]
    return wrapper


class _EntrypointCompiled:
    """Callable wrapper around a compiled 1-node graph."""

    def __init__(self, fn: Callable[[dict], Any], checkpointer: Any | None = None):
        self._fn = fn
        g = _n.StateGraph()

        def node(state: dict) -> dict:
            result = fn(state)
            if isinstance(result, dict):
                return result
            return {"output": result}

        g.add_node("entrypoint", node)
        g.add_edge(START, "entrypoint")
        g.add_edge("entrypoint", END)
        self._graph = g.compile(checkpointer=checkpointer)

    def invoke(self, input: dict, config: dict | None = None) -> Any:
        out = self._graph.invoke(input, config=config)
        return out.get("output", out)

    __call__ = invoke


def entrypoint(*, checkpointer: Any | None = None) -> Callable[[Callable], _EntrypointCompiled]:
    """Decorator factory compiling a function into a native graph.

    Example::

        @entrypoint(checkpointer=MemorySaver())
        def workflow(state: dict) -> dict:
            return {"x": state.get("x", 0) + 1}

        workflow({"x": 1})  # -> {"x": 2}
    """

    def wrap(fn: Callable[[dict], Any]) -> _EntrypointCompiled:
        return _EntrypointCompiled(fn, checkpointer=checkpointer)

    return wrap


__all__ = ["entrypoint", "task"]

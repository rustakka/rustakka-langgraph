"""rustakka_langgraph — native-Rust LangGraph engine with a Python API.

The compiled Rust extension lives in ``rustakka_langgraph._native``; this
module re-exports the public surface under a stable Python-friendly name.
The ``langgraph`` package in this distribution forwards to these symbols to
preserve upstream import paths.
"""

from __future__ import annotations

from . import _native as _n

__version__ = _n.__version__

START: str = _n.START
END: str = _n.END

StateGraph = _n.StateGraph
CompiledStateGraph = _n.CompiledStateGraph
Command = _n.PyCommand if hasattr(_n, "PyCommand") else _n.Command
Send = _n.PySend if hasattr(_n, "PySend") else _n.Send
Interrupt = _n.PyInterrupt if hasattr(_n, "PyInterrupt") else _n.Interrupt

MemorySaver = _n.MemorySaver if hasattr(_n, "MemorySaver") else _n.PyMemorySaver
InMemoryStore = _n.InMemoryStore if hasattr(_n, "InMemoryStore") else _n.PyInMemoryStore

try:
    SqliteSaver = _n.SqliteSaver
except AttributeError:  # feature disabled
    SqliteSaver = None  # type: ignore[assignment]

try:
    PostgresSaver = _n.PostgresSaver
    AsyncPostgresSaver = PostgresSaver
except AttributeError:
    PostgresSaver = None  # type: ignore[assignment]
    AsyncPostgresSaver = None  # type: ignore[assignment]

try:
    PostgresStore = _n.PostgresStore
    AsyncPostgresStore = PostgresStore
except AttributeError:
    PostgresStore = None  # type: ignore[assignment]
    AsyncPostgresStore = None  # type: ignore[assignment]


def interrupt(value):
    """Raise a ``NodeInterrupt``-style signal from within a node.

    Mirrors ``langgraph.types.interrupt``. The current step is persisted by
    the engine; callers resume by invoking the graph with the same config
    and a resume value via :meth:`CompiledStateGraph.resume`.
    """
    return Interrupt(value)


__all__ = [
    "START",
    "END",
    "StateGraph",
    "CompiledStateGraph",
    "Command",
    "Send",
    "Interrupt",
    "MemorySaver",
    "InMemoryStore",
    "SqliteSaver",
    "PostgresSaver",
    "AsyncPostgresSaver",
    "PostgresStore",
    "AsyncPostgresStore",
    "interrupt",
    "__version__",
]

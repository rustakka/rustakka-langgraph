"""``langgraph.checkpoint.sqlite`` — SQLite checkpointer (if compiled)."""

from __future__ import annotations

from rustakka_langgraph import SqliteSaver

if SqliteSaver is None:  # pragma: no cover - feature-gated
    raise ImportError(
        "SqliteSaver is not available: rebuild pylanggraph with the `sqlite` feature."
    )

__all__ = ["SqliteSaver"]

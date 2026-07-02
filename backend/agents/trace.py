from contextvars import ContextVar

_tools: ContextVar[list[str]] = ContextVar("inner_tools", default=[])


def reset() -> None:
    """Start a fresh recording window (call once per question / request)."""
    _tools.set([])


def record(names) -> None:
    """Append one or more inner tool names used by a specialist."""
    _tools.get().extend(names)


def get() -> list[str]:
    """Return the inner tool names recorded since the last reset()."""
    return list(_tools.get())

from contextvars import ContextVar

_tools: ContextVar[list[str]] = ContextVar("inner_tools", default=[])


def reset() -> None:
    _tools.set([])


def record(names) -> None:
    _tools.get().extend(names)


def get() -> list[str]:
    return list(_tools.get())

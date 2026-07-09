import logging
from collections import Counter
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from agents.specialists import (
    cheminformatics_agent,
    safety_agent,
    literature_agent,
    graph_agent,
    molecular_design_agent,
    warmup_subagents,
)
from agents.prompts import (
    SUPERVISOR_SYSTEM_PROMPT as _SUPERVISOR_SYSTEM_PROMPT_TEXT,
    FORCE_ROUTE_DIRECTIVE,
    STOP_AND_ANSWER_DIRECTIVE,
)
from core.config import OLLAMA_BASE_URL, MANAGER_MODEL

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [cheminformatics_agent, safety_agent, literature_agent, graph_agent, molecular_design_agent]
_SPECIALIST_NAMES = {t.name for t in tools}

# Max times a specialist may be called before forcing a text-only answer.
_MAX_SPECIALIST_CALLS = 2

def _surface_specialist_error(exc: Exception) -> str:
    return (
        f"SPECIALIST_ERROR: {exc}. This specialist produced NO result. Report this "
        f"failure to the user plainly; do NOT fabricate, guess, or work around the "
        f"missing result."
    )

tool_node = ToolNode(tools=tools, handle_tool_errors=_surface_specialist_error)
_llm_supervisor = None
_llm_supervisor_with_tools = None
orchestrator = None

# Retry directive for first-hop answers without routing.
# Some models skip required tool calls; since Ollama ignores tool_choice,
# we re-prompt once to enforce routing.
_FORCE_ROUTE_DIRECTIVE = SystemMessage(content=FORCE_ROUTE_DIRECTIVE)

def warmup():
    try:
        _llm_supervisor.invoke("ready?")
    except Exception:
        logger.warning("Supervisor model warmup skipped", exc_info=True)
    warmup_subagents()

SUPERVISOR_SYSTEM_PROMPT = SystemMessage(content=_SUPERVISOR_SYSTEM_PROMPT_TEXT)

_STOP_AND_ANSWER_DIRECTIVE = SystemMessage(content=STOP_AND_ANSWER_DIRECTIVE)


def supervisor_node(state: AgentState):
    messages = [SUPERVISOR_SYSTEM_PROMPT] + state["messages"]
    routed = any(isinstance(m, ToolMessage) for m in state["messages"])
    call_counts = Counter(
        m.name for m in state["messages"]
        if isinstance(m, ToolMessage) and m.name in _SPECIALIST_NAMES
    )
    if call_counts and max(call_counts.values()) >= _MAX_SPECIALIST_CALLS:
        response = _llm_supervisor.invoke(messages + [_STOP_AND_ANSWER_DIRECTIVE])
        return {"messages": [response]}
    response = _llm_supervisor_with_tools.invoke(messages)
    if not routed and not getattr(response, "tool_calls", None):
        response = _llm_supervisor_with_tools.invoke(messages + [_FORCE_ROUTE_DIRECTIVE])
    return {"messages": [response]}


def build_orchestrator() -> None:
    global _llm_supervisor, _llm_supervisor_with_tools, orchestrator

    _llm_supervisor = ChatOllama(
        model=MANAGER_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=8192,
        num_predict=1024,
        keep_alive=-1,
        reasoning=False
    )
    _llm_supervisor_with_tools = _llm_supervisor.bind_tools(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", tools_condition)
    workflow.add_edge("tools", "supervisor")
    orchestrator = workflow.compile()

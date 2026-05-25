from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

from tools.pubmed_api import search_pubmed
from tools.pubchem_api import fetch_pubchem_properties
from tools.toxicity_predictor import predict_toxicity
from rag.retriever import search_literature

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [search_pubmed, fetch_pubchem_properties, search_literature, predict_toxicity]
tool_node = ToolNode(tools=tools)

llm_manager = ChatOllama(model="llama3.1", base_url="http://host.docker.internal:11434", temperature=0)
llm_manager_with_tools = llm_manager.bind_tools(tools)

llm_scientist = ChatOllama(model="qwen2.5", base_url="http://host.docker.internal:11434", temperature=0.3)

def manager_node(state: AgentState):
    """Main node that decides whether to call a tool or generate a response."""
    response = llm_manager_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def scientist_node(state: AgentState):
    """Node that generates the final response for the user."""

    strict_prompt = """You are a strict output formatting agent. Your ONLY job is to take the data from the tools and write a final clean response.

CRITICAL RULES FOR CHEMICAL FORMULAS:
- If you write a SMILES formula (like c1ccccc1, CN1C=NC2=..., etc.), you MUST wrap it in <smiles> and </smiles>.
- NEVER use quotes, single quotes, or backticks around a SMILES string.
- EXAMPLE OF CORRECT OUTPUT: The generated derivative is <smiles>CC(=O)OC1=CC=CC=C1C(=O)O</smiles> which has a low toxicity risk.

CRITICAL RULES FOR NUMBERS AND SCORES:
- When printing toxicity probabilities or safety scores (like 0.85, 99.00%, 0.35), print them as normal text. 
- NEVER wrap plain numbers, scores, or percentages in <smiles> tags or quotes. 

Do not log steps, do not output raw JSON code blocks."""

    messages = state["messages"] + [HumanMessage(content=strict_prompt)]
    response = llm_scientist.invoke(messages)
    return {"messages": [response]}

def route_after_manager(state: AgentState):
    """Route to tool node if the manager decides to call a tool, otherwise route to scientist."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "scientist"

workflow = StateGraph(AgentState)

workflow.add_node("manager", manager_node)
workflow.add_node("scientist", scientist_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "manager")

workflow.add_conditional_edges(
    "manager",
    route_after_manager,
    {
        "tools": "tools",
        "scientist": "scientist"
    }
)

workflow.add_edge("tools", "manager")
workflow.add_edge("scientist", END)

orchestrator = workflow.compile()
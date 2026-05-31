from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama

from tools.pubmed_api import search_pubmed
from tools.pubchem_api import fetch_pubchem_properties
from tools.toxicity_predictor import predict_toxicity
from tools.smiles_validator import validate_smiles
from tools.property_calculator import calculate_properties
from rag.retriever import search_literature

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [search_pubmed, fetch_pubchem_properties, search_literature, predict_toxicity, validate_smiles, calculate_properties]
tool_node = ToolNode(tools=tools)

llm_manager = ChatOllama(model="qwen2.5", base_url="http://host.docker.internal:11434", temperature=0)
llm_manager_with_tools = llm_manager.bind_tools(tools)

MANAGER_SYSTEM_PROMPT = SystemMessage(content="""You are a drug discovery research assistant that coordinates tools to answer scientific questions.

TOOL ORDERING RULES — follow these strictly:
1. Only call tools that are needed to answer the user's question. Do not call extra tools the user did not ask for.
2. If the user asks for toxicity prediction of a known drug, you MUST call fetch_pubchem_properties first to get the SMILES, then call predict_toxicity in a separate step with the SMILES from the result. Never call both in the same step.
3. search_pubmed and search_literature are independent and can be called alongside other tools in any step.
4. Do NOT call the same tool with the same input more than once per conversation.

SMILES RULES — critical:
- After fetch_pubchem_properties returns, find the line that starts with "- SMILES:" in the result. The value after the colon is the SMILES string. Example line: "- SMILES: CC(=O)Nc1ccc(O)cc1". Pass that exact string to predict_toxicity with no modifications.
- NEVER generate, guess, invent, or reconstruct a SMILES string from memory.
- Tool arguments must be plain SMILES strings only — no tags, no quotes, no markup of any kind.
- If predict_toxicity returns an Invalid SMILES error, go back to the fetch_pubchem_properties result already in this conversation and copy the SMILES from there exactly. Do not create a new SMILES.

OUTPUT FORMATTING RULES — only for your final text response, never for tool arguments:
- Wrap every SMILES string in <smiles> and </smiles> tags. Example: <smiles>CC(=O)Nc1ccc(O)cc1</smiles>
- Never put quotes, backticks, or single quotes around a SMILES string.
- Print toxicity scores and percentages as plain text. Never wrap numbers in <smiles> tags.
- Do not output raw JSON or log intermediate steps.

Always use real tool calls — never describe or narrate what you would call.""")

def manager_node(state: AgentState):
    messages = [MANAGER_SYSTEM_PROMPT] + state["messages"]
    response = llm_manager_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)

workflow.add_node("manager", manager_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "manager")
workflow.add_conditional_edges("manager", tools_condition)
workflow.add_edge("tools", "manager")

orchestrator = workflow.compile()

import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from tools.pubmed_api import search_pubmed
from tools.pubchem_api import fetch_pubchem_properties
from rag.retriever import search_literature

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [search_pubmed, fetch_pubchem_properties, search_literature]

llm = ChatOllama(model="llama3.1", base_url="http://ollama:11434")
# llm = ChatOllama(model="llama3.1", base_url="http://host.docker.internal:11434")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(AgentState)

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge("tools", "chatbot")

orchestrator = graph_builder.compile()

if __name__ == "__main__":
    while True:
        user_input = input("Prompt: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        print("\Thinking...")
        start_time = time.time()
        events = orchestrator.stream(
            {"messages": [("user", user_input)]}, stream_mode="values"
        )
        
        for event in events:
            event["messages"][-1].pretty_print()
            
        end_time = time.time()
        print(f"\nWaiting time: {end_time - start_time:.2f} seconds")
        print("-" * 50)
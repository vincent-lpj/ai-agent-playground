import asyncio
import operator
from typing import TypedDict, Annotated

from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient

# Config
LLM_MODEL = "qwen3:8b"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)

# Define the state for the agent
class AgentState(TypedDict):
    messages: Annotated[list[SystemMessage | HumanMessage], operator.add]


# Function to load tools from MCP servers
async def get_tools():
    mcp_client = MultiServerMCPClient(
        connections={
            "airbnb": {
                "command": "npx",
                "args": [
                    "-y",
                    "@openbnb/mcp-server-airbnb",
                    "--ignore-robots-txt"
                    ],
                "transport": "stdio"
                },
        }
    )

    tools = await mcp_client.get_tools()
    print(f"{len(tools)} tools loaded from Airbnb MCP servers.")
    return tools


# Agent node that uses the tools to generate responses
async def agent_node(state: AgentState):
    TOOLS = await get_tools()
    llm_with_tools = llm.bind_tools(tools=TOOLS)

    system_message = SystemMessage("You are a helpful assistant that can use tools from the Airbnb MCP servers to answer user queries.")
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)

    # AIMessages will contain tool_calls attributes during tool call turn
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            print(f"This agent called tool: {tc.get('name', '?')} with args: {tc.get('args', '?')}")
    else:
        print(f"[AGENT] generating responses...")

    return {"messages": [response]}


# Build the agent graph
async def create_agent():
    TOOLS = await get_tools()

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools=TOOLS))

    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")

    # Note here tools_condition will automatically routes when calling tools
    builder.add_conditional_edges("agent", tools_condition)

    graph = builder.compile()

    return graph


# Example of running the agent with user input
async def search(query:str):
    agent = await create_agent()
    state = AgentState(messages=[HumanMessage(query)])
    response = await agent.ainvoke(input=state)
    response["messages"][-1].pretty_print()


# Main loop for chatting with the agent
if __name__ == "__main__":
    while True:
        query = input("Enter your query (or '/exit' to quit): ")
        if query.lower() == "/exit":
            break
        asyncio.run(search(query))
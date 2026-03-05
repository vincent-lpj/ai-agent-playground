import os
import re
import operator
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from langgraph.types import Command, interrupt

# Use Postgres
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore


# =============================================
# Configuration
# =============================================

# Load environment variables from .env file, so that LangSmith tracing is enabled.
load_dotenv()

# Configuration for Ollama server and model
BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
POSTGRES_CONNINFO = os.getenv("POSTGRES_CONNINFO")

# Initialize the LLM and embeddings
llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL)

# Define the state schema for the agent. The state will keep track of the conversation messages.
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    is_sensitive: bool  # flag to indicate if the last message contains sensitive information


# =============================================
# Store and Saver Setup
# =============================================
# Define a simple embedding function
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)

# Set up memory saver and store
def setup_memory(postgres_conninfo: str):
    # Short term memory saver using Postgres, for checkpointing the agent state and conversation history
    checkpointer_conn = psycopg.connect(conninfo=postgres_conninfo, autocommit=True, prepare_threshold=0)
    checkpointer = PostgresSaver(checkpointer_conn) 
    checkpointer.setup()     # first time setup the database tables for the store

    # Long term memory store and saver using Postgres
    store_conn = psycopg.connect(conninfo=postgres_conninfo, autocommit=True, prepare_threshold=0)
    store = PostgresStore(store_conn, index={'embed': embed_texts, 'dims': 768}) # embedding dimension is 768 for nomic-embed-text
    store.setup()           # first time setup the database tables for the store

    return checkpointer, store

checkpointer, store = setup_memory(POSTGRES_CONNINFO)

# =============================================
# Transfer Money Tool
# =============================================
@tool
def transfer_money(amount: int, recipient: str) -> dict:
    """
    Transfer moneny. Large transfers require approval

    Args:
        amount (int): Amount to transfer in dollars
        recipient (str): Recipient of the transfer
    """
    if amount > 1000:
        approval = interrupt(
            {
                "type": "approval_required",
                "amount": amount,
                "recipient": recipient
            }
        )
        if approval.get("decision") != "approve":
            return {"status": "canceled", "amount": amount, "recipient": recipient}
        
    return {"status": "success", "amount": amount, "recipient": recipient}

# =============================================
# Tools for Memory Management
# =============================================

@tool
def get_user_memory(user_id: str, category: str) -> dict:
    """
    Retrieve user preference or information from long term memory

    Args:
        user_id: User identifier
        category: Category of information (e.g. 'food', 'hobbies', 'schedule', 'location)
    """
    namespace = (user_id, "preference")

    try:
        item = store.get(namespace=namespace, key=category)
        return {"status": "succeeded",
                "error": "N/A",
                "data_retrieved": item.value}
    except Exception as e:
        return {"status": "failed",
                "error": e,
                "data_retrieved": "Not Found"}


@tool
def save_user_memory(user_id: str, category: str, information: dict) -> dict:
    """
    Save user preference or information to long-term memory.

    Args:
        user_id: User identifier
        category: Category of information (e.g. 'food', 'hobbies', 'schedule', 'location)
        information: Dictionary that containing the information to use
    """
    namespace = (user_id, "preferrence")
    try:
        store.put(namespace=namespace, key=category, value=information)
    except Exception as e:
        return {"status": "failed",
                "error": e}
    return {"status": "succeeded", "error": "N/A"}


# =============================================
# General Tools
# =============================================

# Define a tool for getting weather information. 
@tool
def get_weather(location: str) -> dict:
    """Get current weather for a location.
    
    Use for queries about weather, temperature, or conditions in any city.
    Examples: "weather in Paris", "temperature in Tokyo", "is it raining in London"
    
    Args:
        location: City name (e.g., "New York", "London", "Tokyo")
        
    Returns:
        Current weather information including temperature and conditions.
    """

    # In a real implementation, you would call an actual weather API here.
    # url = f"https://wttr.in/{location}?format=j1"
    # response = requests.get(url=url, timeout=10)
    # response.raise_for_status()
    # data = response.json()
    
    # hard-coding because wttr.in is down
    weather_data = {
        "location": location,
        "weather": "rainy, but rain will stop in 2 hours."}

    return weather_data

# Define the list of tools that the agent can use. This will be passed to the ToolNode in the graph.
TOOLS = [get_user_memory, save_user_memory, get_weather, transfer_money]


# =============================================
# Guardrail Node for PII Detection
# =============================================

# PII Pattern Definitions
patterns = {
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',  # SSN: 123-45-6789
        "Credit Card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit Card: 1234-5678-9012-3456
        "Mobile Number": r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Mobile: +1-234-567-8900
        "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email: user@example.com
        "URL/Link": r'https?://[^\s]+|www\.[^\s]+'  # URL: http://example.com or www.example.com
    }

def guardrail_node(state: AgentState):
    last_message = state["messages"][-1].content

    for pii_type, pii_pattern in patterns.items():
        if re.search(pattern=pii_pattern, string=last_message):
            print(f"PII detected. Type {pii_type}")
            # Will be connected to the END node and share with user
            return {
                "messages": [AIMessage(content=f"Request Blocked: Contains {pii_type}.\nPlease Do not share sensitive personal information to the agent")],
                "is_sensitive": True
            }
    else:
        return {
            "is_sensitive": False,
        }


# =============================================
# Agent Node with Long-Term and Short-Term Memory
# =============================================
def agent_node(state: AgentState):
    # Bind tools to the LLM, so that it can call tools during generation.
    llm_with_tools = llm.bind_tools(tools=TOOLS)

    user_id = state.get("user_id", "unknown")
    namespace = (user_id, "preference")

    last_message = state["messages"][-1].content
    memories = store.search(namespace, query=last_message, limit=3)

    # build context memory for personalized answer
    context_line = []
    for mem in memories:
        text = f" - {mem.key}: {mem.value}"
        context_line.append(text)
    memory_text = "\n\n".join(context_line) if context_line else "No user preference found in the store yet."
    
    SYSTEM_PROMPT = f"""
                    You are a helpful assistant with long-term memory capabilities and access to utility tools.

                        User ID: {user_id}
                        Current User Memories:
                        {memory_text}

                        MEMORY TOOLS USAGE:

                        1. save_user_memory: Use when user shares NEW information
                        - Always pass user_id: "{user_id}"
                        - Food preferences (diet, likes, dislikes, allergies)
                        - Work information (role, company, interests)
                        - Hobbies and activities
                        - Schedule and availability
                        - Location and timezone

                        2. get_user_memory: Use when you need to recall specific category
                        - Always pass user_id: "{user_id}"
                        - When answering questions about past preferences
                        - When user asks "what do you know about me?"
                        - When making recommendations based on preferences

                        UTILITY TOOLS USAGE:                            
                        3. get_weather: Use to retrieve current weather information
                        - Pass location as parameter (city name, zip code, or coordinates)
                        - Use when user asks about weather conditions
                        - Use when planning activities that depend on weather
                        - Examples: "What's the weather in London?", "Will it rain today?"

                        4. transfer_money: Use to transfer money. Large transfers over $1000 require approval.
                        - Pass amount and recipient as parameters
                        - Use when user asks to transfer money
                        - Examples: "Send $1500 to Bob"

                        GUIDELINES:
                        - Always save when user shares personal information
                        - Retrieve specific categories when needed for context
                        - Use semantic search results shown above for general context
                        - Use get_weather when location-based weather info is needed
                        - Use calculate for any mathematical operations or conversions
                        - Be conversational and natural when using all tools
                        - Combine tools when appropriate (e.g., weather + saved location preference)
                    """

    system_message = SystemMessage(SYSTEM_PROMPT)
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)

    # AIMessages will contain tool_calls attributes during tool call turn
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            print(f"This agent called tool: {tc.get('name', '?')} with args: {tc.get('args', '?')}")
    else:
        print(f"[AGENT] generating responses...")

    # The value of messages should be list, because of operator.add
    return {"messages": [response]}

# =============================================
# Conditional Edge
# =============================================

# Guardrail, agent
def guardrail_router(state: AgentState):
    is_sensitive = state["is_sensitive"]
    
    if is_sensitive:
        return END
    else:
        return "agent"

# Define the routing function
def should_continue(state:AgentState):
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"      # This key should be the same with tool node
    else:
        return END

# =============================================
# Graph
# =============================================

# Build the agent graph
def create_agent(checkpointer):
    # Get graph canvas
    # The canvas should accept state schema, instead of state instance
    builder = StateGraph(state_schema=AgentState)

    builder.add_node("guardrail", guardrail_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools=TOOLS)) # Should be the same with that in routing 

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(source="guardrail", path=guardrail_router, path_map=["agent", END])
    builder.add_conditional_edges(source="agent", path=should_continue, path_map=["tools", END])
    builder.add_edge("tools", "agent")  # Do not forget to connect tool node back to agent

    graph = builder.compile(checkpointer=checkpointer)

    return graph

# Define chat function to interact with the agent
def chat(agent, query, user_id, thread_id) -> AgentState:
    initial_state = AgentState(
        messages=[HumanMessage(query)],
        user_id=user_id
        )

    # Configure thread id 
    config = {"configurable" :{"thread_id": thread_id}}

    response = agent.invoke(initial_state, config=config)
    return response


# =============================================
# Main Loop for Chatting with the Agent
# =============================================

# Create the agent and chat with it
user_id = "demo-user-001"
thread_id = "agent-with-memory-HITL-001"
agent = create_agent(checkpointer=checkpointer) 


# main loop for chatting with the agent
if __name__ == "__main__":
    while True:
        query = input("Enter your query (or '/exit' to quit): ")
        if query.lower() == "/exit":
            break
        response = chat(agent, query, user_id, thread_id)
        if "__interrupt__" in response:
            approval_info = response["__interrupt__"][0].value
            print(f"Approval required: You are sending {approval_info.get('amount', 'unknown')} to {approval_info.get('recipient', 'unknown')}.")
            approval = input("Do you approve this move (enter 'approve' or 'disapprove'): ")
            execution = agent.invoke(input=Command(resume={"decision": approval}), config={"configurable" :{"thread_id": thread_id}})
            execution["messages"][-1].pretty_print()
        else:
            response["messages"][-1].pretty_print()
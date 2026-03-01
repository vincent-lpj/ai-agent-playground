from langchain_ollama import ChatOllama
from langchain.agents import create_agent

llm = ChatOllama(
    model="qwen3:8b"
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
# Returns a dictionary with "messages" key
# Within messages, a list of HumanMessage/AIMEssage is stored
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

response["messages"][-1].pretty_print()
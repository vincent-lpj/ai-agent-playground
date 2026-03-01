from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    model="qwen3:8b"
)

messages = [SystemMessage(content = "You are a friendly and helpful assistent."), 
            HumanMessage(content="Tell me about AI agent")]

response = llm.invoke(input=messages)
response.pretty_print()


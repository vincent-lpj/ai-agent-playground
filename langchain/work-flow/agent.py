import operator
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

@tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

def llm_call(state: AgentState):
    llm = ChatOllama(model="qwen3:8b")
    llm_with_tools = llm.bind_tools(tools=[get_weather])

    messages = state["messages"]
    llm_input = [SystemMessage(content="You are a helpful and friendly assistent. Try to use your tool to search for weather infos")] + messages

    response = llm_with_tools.invoke(input=llm_input)

    return {"messages": [response], "llm_calls": state["llm_calls"] + 1}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_call"
    
    return END


def create_workflow(state_schema: AgentState):
    builder = StateGraph(state_schema=state_schema)
    
    builder.add_node(node="llm_call", action=llm_call)
    builder.add_node(node="tool_call", action=ToolNode(tools=[get_weather]))
    builder.add_edge(start_key=START, end_key="llm_call")
    builder.add_conditional_edges(source="llm_call", path=should_continue, path_map=["tool_call", END])
    builder.add_edge(start_key="tool_call", end_key="llm_call")
    
    work_flow = builder.compile()

    return work_flow

initial_state = AgentState(messages=[HumanMessage("Please tell me today's weather in new york")], llm_calls=0)
work_flow = create_workflow(state_schema=AgentState)

state = work_flow.invoke(initial_state)

state["messages"][-1].pretty_print()
print(f"Total LLM calls: {state["llm_calls"]}")




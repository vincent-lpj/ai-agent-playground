from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from datetime import datetime


from dotenv import load_dotenv

load_dotenv()

llm_ollama = LiteLlm(
        model="ollama_chat/qwen3:8b"
    )

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

def get_current_time() -> dict:
    """Retrieves the current time.

    Args:
        N/A

    Returns:
        dict: status and result or error msg.
    """
    try:
        current_time_str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        return {
            "status": "succeed",
            "current_time": current_time_str
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "current_time": "unknown"
        }


weather_agent = Agent(
    name="weather_agent",
    model=llm_ollama,
    description="Agent to answer questions about the weather of a specific city.",
    instruction="""
                You are a helpful agent who can answer user questions about the time and weather in a city.
                Make sure to use this tool: get_weather to get the most recent infos about current weather
                """,
    tools=[get_weather]
)

time_report_agent = Agent(
    name="time_report_agent",
    model=llm_ollama,
    description="Agent to answer questions about the current time",
    instruction="""
                You are a helpful agent who can answer user questions about the current time.
                Make sure to use this tool: get_current_time to get the most accurate time.
                """,
    tools=[get_current_time]
)

root_agent = Agent(
    name='root_agent',
    model= llm_ollama,
    description=(
        "Root agent"
    ),
    instruction="""
        You are a helpful agent. Delegate tasks to proper sub-agent.",
        "You have two sub-agents: weather_agent for answering question about a city's weather and time_report_agent who reports current time.
        """,
    sub_agents=[weather_agent, time_report_agent]
)

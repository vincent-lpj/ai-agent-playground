from autogen_ext.models.ollama import OllamaChatCompletionClient
import asyncio
from autogen_agentchat.agents import AssistantAgent

# Define a tool that searches the web for information.
# For simplicity, we will use a mock function here that returns a static string.
async def get_weather(city: str) -> dict:
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


# Create an agent that uses the Ollma qwen3 model.
ollama_model_client = OllamaChatCompletionClient(model="qwen3:8b")


agent = AssistantAgent(
    name="assistant",
    model_client=ollama_model_client,
    tools=[get_weather],
    system_message="Use tools to solve tasks.",
    max_tool_iterations=3
)

async def main():
    # The call to the run() method returns a TaskResult with the list of messages in the messages attribute, 
    # which stores the agent’s “thought process” as well as the final response.
    result = await agent.run(task="Find information on new york weather")
    # for env in result.messages:
    #     print("\n", env, "\n")
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())

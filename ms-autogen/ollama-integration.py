from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
import asyncio

# Assuming your Ollama server is running locally on port 11434.
ollama_model_client = OllamaChatCompletionClient(model="qwen3:8b")


async def main():
    response = await ollama_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(response)
    await ollama_model_client.close()

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from lib.infernece import generate_text

async def chat_with_model(model: str):
    print(f"Chat with model: {model}")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        
        response = await generate_text(model, prompt)
        print(f"Model: {response}")

if __name__ == "__main__":
    model_input = input("Enter the model name (e.g., gpt-3.5-turbo): ")
    asyncio.run(chat_with_model(model_input))

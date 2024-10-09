import asyncio
from lib.infernece import generate_text
from lib.classify import classify_text

MODEL_CONFIGS = {
    "small": {
        "Math": "Qwen/Qwen2-Math-7B",
        "STEM": "Qwen/Qwen2-7B",
        "Coding": "meta-llama/Llama-2-8b-chat",
        "Other": "google/gemma-2b-it"
    },
    "medium": {
        "Finance": "palmyra-finance/palmyra-finance-72b",
        "Health": "palmyra-health/palmyra-health-72b",
        "Math": "Qwen/Qwen2-72B-Math-Instruct",
        "STEM": "Qwen/Qwen2-72B-Instruct",
        "Coding": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "Other": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    },
    "large": {
        "Finance": "claude-3-5-sonnet-20240620",
        "Health": "claude-3-5-sonnet-20240620",
        "Math": "gpt-4o",
        "STEM": "claude-3-5-sonnet-20240620",
        "Coding": "claude-3-5-sonnet-20240620",
        "Other": "gpt-4o"
    }
}

async def route_and_infer(prompt, model_size="medium"):
    # Classify the prompt
    category = classify_text(prompt)

    print(f"Classified as: {category}")

    # Check if the model_size exists in MODEL_CONFIGS
   
    # Get the appropriate model based on the category and model size
    if category in MODEL_CONFIGS[model_size]:
        model = MODEL_CONFIGS[model_size][category]
    else:
        model = MODEL_CONFIGS[model_size]["Other"]

    print(f"Routing to model: {model}")

    if category == "Math" and model_size == "medium":
        print("Math category")
        print(prompt)
        print('\n'*3)

    # Add CoT instruction for Math and STEM categories
    if category in ["Math", "STEM"]:
        prompt = "Use Chain of Thought reasoning to solve this problem. Show your step-by-step thinking process: " + prompt

    # Generate text using the selected model
    try:
        output = await generate_text(model, prompt)
        return output
    except Exception as e:
        print(f"Error with {model}: {str(e)}")
        return None

async def main():
    model_size = input("Enter model size (small/medium/large): ").lower()
    while model_size not in ["small", "medium", "large"]:
        model_size = input("Invalid size. Please enter small, medium, or large: ").lower()

    prompt = input("Enter your prompt: ")
    
    result = await route_and_infer(prompt, model_size)
    if result:
        print("\nGenerated text:")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())

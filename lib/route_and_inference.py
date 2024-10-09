import asyncio
from lib.infernece import generate_text
from lib.classify import classify_text
from collections import Counter

MODEL_CONFIGS = {


    #Actual small
    # "small": { #<9B
    #     "Health": "Qwen/Qwen2-7B-Instruct",
    #     "Math": "Qwen/Qwen2-Math-72B-Instruct",
    #     "STEM": "Qwen/Qwen2-7B-Instruct",
    #     "Coding": "open-codestral-mamba",
    #     "Other": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # },

    #Test small
    "small": { #<9B
        "Health": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Math": "mistralai/mathstral-7b-v0.1",
        "STEM": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Coding": "open-codestral-mamba",
        "Other": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    },
   
#    "medium": {
#         "Finance": "palmyra-finance/palmyra-finance-72b",
#         "Health": "palmyra-health/palmyra-health-72b",
#         "Math": "Qwen/Qwen2-72B-Instruct",
#         "STEM": "Qwen/Qwen2-Math-72B-Instruct",
#         "Coding": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
#         "Other": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
#     },

    #Medium model
    "medium": {
        "Finance": "palmyra-health/palmyra-health-72b",
        "Health": "palmyra-health/palmyra-health-72b",
        "Math": "Qwen/Qwen2-Math-72B-Instruct",
        "STEM": "Qwen/Qwen2-72B-Instruct",
        "Coding": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "Other": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    },
    "large": {
        
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

    print(f"\nClassified as: {category}\n")

    # Check if the model_size exists in MODEL_CONFIGS
   
    # Get the appropriate model based on the category and model size
    if category in MODEL_CONFIGS[model_size]:
        model = MODEL_CONFIGS[model_size][category]
    else:
        model = MODEL_CONFIGS[model_size]["Other"]

    #print(f"Routing to model: {model}")

    

    # Add CoT instruction for Math and STEM categories
    if category != "Other":
        prompt = f'You are an expert in {category}.' + prompt
    # Generate text using the selected model
    try:
        output = await generate_text(model, prompt)
        return output
    except Exception as e:
        print(f"Error with {model}: {str(e)}")
        return None

# async def main():
#     model_size = input("Enter model size (small/medium/large): ").lower()
#     while model_size not in ["small", "medium", "large"]:
#         model_size = input("Invalid size. Please enter small, medium, or large: ").lower()

#     classifications = Counter()

#     while True:
#         prompt = input("Enter your prompt (or 'q' to quit): ")
#         if prompt.lower() == 'q':
#             break

#         category = classify_text(prompt)
#         classifications[category] += 1
        
#         result = await route_and_infer(prompt, model_size)
#         if result:
#             print("\nGenerated text:")
#             print(result)
        
#         print("\n" + "-"*40 + "\n")

#     print("\nClassification Summary:")
#     for category, count in classifications.items():
#         print(f"{category}: {count}")

# if __name__ == "__main__":
#     asyncio.run(main())

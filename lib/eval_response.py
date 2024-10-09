import os
from together import AsyncTogether

# Load the API key from an environment variable
together_api_key = os.environ.get("TOGETHER_API_KEY")

# Initialize the AsyncTogether client with the environment variable
async_together_client = AsyncTogether(api_key=together_api_key)

async def evaluate_text(eval_model: str, modelAnswer: str, groundTruthAnswer: int = 1000, temperature: float = 0) -> str:
    isAccurate = await async_together_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You will be given a ground truth answer and a model answer. Please output ACCURATE if the model answer matches the ground truth answer or INACCURATE otherwise. Please only return ACCURATE or INACCURATE. It is very important for my job that you do this.",
                },
                {
                    "role": "user",
                    "content": f"""
                        <GroundTruthAnswer>
                        {groundTruthAnswer}
                        </GroundTruthAnswer>

                        <ModelAnswer>
                        {modelAnswer}
                        </ModelAnswer>
                        """,
                },
            ],
            model=eval_model,
        )
    if isAccurate.choices[0].message.content == "ACCURATE":
        return 1, 0
    elif isAccurate.choices[0].message.content == "INACCURATE":
        return 0, 0
    else:
        return 0, 1  # Return 1 for badResponses


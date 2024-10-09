import os
import openai
import anthropic
import aiohttp
import json
from together import AsyncTogether
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import asyncio
import requests
from mistralai import Mistral

# Load environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
together_api_key = os.environ.get("TOGETHER_API_KEY")
writer_api_key = os.environ.get("WRITER_API_KEY")
hf_api_token = os.environ.get("HF_API_TOKEN")
mistral_key = os.environ.get("MISTRAL_KEY")
nvidia_key = os.environ.get("NVIDIA_KEY")

#hf_client = InferenceClient(token=hf_api_token)  # Replace with your Hugging Face token

# Initialize clients
async_together_client = AsyncTogether(api_key=together_api_key)
async_openai_client = AsyncOpenAI(api_key=openai_api_key)

async def generate_text(model: str, prompt: str, max_tokens: int = 1000, temperature: float = 0) -> str:
    """
    Asynchronously generate text using various AI models.
    
    :param model: The name of the model to use (e.g., "gpt-3.5-turbo", "claude-2", "meta-llama/Llama-2-70b-chat-hf")
    :param prompt: The input prompt for text generation
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in generation (0.0 to 1.0)
    :return: Generated text as a string
    """
    
    # OpenAI models
    if model.startswith("gpt-"):
        response = await async_openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    # Anthropic (Claude) models
    elif model.startswith("claude-"):
        async def run_anthropic():
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            if model.startswith("claude-3"):
                response = client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.content[0].text.strip()
            else:
                response = client.completions.create(
                    model=model,
                    prompt=f"Human: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                return response.completion.strip()
        
        return await run_anthropic()
    
    # Together AI models
    elif model.startswith("mistral") or model.startswith("open-"):
        if model == "mistralai/mathstral-7b-v0.1":  # New condition for specific model
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key  # Use the nvidia_key variable
            )
            
            # Update for handling non-streamed response
            completion = client.chat.completions.create(  # No await here
                model="mistralai/mathstral-7b-v0.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=1024
            )
            
            # Return the non-streamed response
            return completion.choices[0].message.content
        
        # Existing Mistral code remains unchanged
        client = Mistral(api_key=mistral_key)
        chat_response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return chat_response.choices[0].message.content
    
    elif any(model.startswith(prefix) for prefix in ["meta-llama/",  "togethercomputer/", "google", "Qwen/", "Meta-Llama"]):
        prompt_formatted = [
            {"role": "user", "content": prompt}
        ]
        
        response = await async_together_client.chat.completions.create(
            model=model,
            messages=prompt_formatted,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    # Writer AI models
    elif model.startswith("palmyra"):
        async with aiohttp.ClientSession() as session:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {writer_api_key}"
            }
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            async with session.post("https://api.writer.com/v1/completions", headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['text'].strip()
                else:
                    raise Exception(f"Error with Writer AI: {await response.text()}")
    
    elif model.startswith("small") or model.startswith("medium") or model.startswith("large"):
        from lib.route_and_inference import route_and_infer  # Import here
        return await route_and_infer(prompt, model)
    
    # Hugging Face models (fallback for any other model)
    else:
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
        async def query_huggingface():
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                        else:
                            return str(result)
                    else:
                        raise Exception(f"Error with Hugging Face Inference API for model {model}: {await response.text()}")
        
        return await query_huggingface()
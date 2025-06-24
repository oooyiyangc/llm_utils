from openai import OpenAI
import os
import warnings
import json
import pydantic

from tqdm import tqdm
from typing import List, Tuple, Optional

OPENAI_MODEL_ALIAS = {
    "gpt-3.5": "gpt-3.5-turbo-0125", 
    "gpt-4o": "gpt-4o", 
    "gpt-4o-mini": "gpt-4o-mini"
}
MAX_TOKENS = 4096

def create_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client

    Args:
        api_key (str): OpenAI API key

    Returns:
        OpenAI: OpenAI client
    """

    if api_key is None and "OPENAI_API_KEY" not in os.environ:
        raise ValueError("API key is required")
    
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]
    
    client = OpenAI(api_key=api_key, timeout=120)

    return client


def get_response(client: OpenAI, user_input: str, system_prompt: Optional[str] = None, model: str = "gpt-4o-mini", max_tokens: int = 1024, temperature: float = 0, top_p: float = 1, response_format: Optional[pydantic.BaseModel] = None) -> str:
    """
    Get response from OpenAI API
    
    Args:
        client (OpenAI): OpenAI client
        user_input (str): User input
        system_prompt (str): System prompt
        model (str): Model to use
        max_tokens (int): Maximum number of tokens
        temperature (float): Temperature for sampling
        
    Returns:
        str: Response from OpenAI API
    """

    if model in OPENAI_MODEL_ALIAS:
        model = OPENAI_MODEL_ALIAS[model]
    
    # print(f"Getting response with model {model}...")

    if system_prompt is not None:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    else:
        warnings.warn("No system prompt provided. Using default system prompt.")
        messages = [
            {
                "role": "user",
                "content": user_input
            }
        ]

    if response_format is not None:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=min(max_tokens, MAX_TOKENS),
            top_p=top_p,  
            frequency_penalty=0,
            presence_penalty=0,
            seed=42, 
            response_format=response_format
        )
        return response.choices[0].message.parsed

    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=min(max_tokens, MAX_TOKENS),
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            seed=42
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    
    OPENAI_API_KEY = "sk-..."
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    print("Creating client...")
    client = create_client()

    # test_input_1 = "Hello"
    # print(get_response(client, test_input_1))
import anthropic
import os
import warnings
import json

from tqdm import tqdm

CLAUDE_MODEL_ALIAS = {
    "opus": "claude-3-opus-20240229",
    "sonnet": "claude-3-sonnet-20240229",
    "haiku": "claude-3-haiku-20240307"
}
MAX_TOKENS = 4096

def create_client(api_key=None):
    if api_key is None and "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError("API key is required")
    
    if api_key is None:
        api_key = os.environ["ANTHROPIC_API_KEY"]
    
    client = anthropic.Anthropic(
        api_key=api_key
    )
    return client


def get_response(client, user_input, system_prompt=None, model="haiku", max_tokens=1024, temperature=0):

    if model in CLAUDE_MODEL_ALIAS:
        model = CLAUDE_MODEL_ALIAS[model]
    
    # print(f"Getting response with model {model}...")

    if system_prompt is None:
        warnings.warn("No system prompt provided. Using default system prompt.")
        message = client.messages.create(
            model=model,
            max_tokens=min(max_tokens, MAX_TOKENS),
            messages=[
                {"role": "user", "content": user_input}
            ], 
            temperature=temperature
        )   

    else:
        message = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=min(max_tokens, MAX_TOKENS),
            messages=[
                {"role": "user", "content": user_input}
            ], 
            temperature=temperature
        )
    return message.content[0].text

if __name__ == "__main__":

    api_key = "sk-ant-api03-..."

    print("Creating client...")
    client = create_client(api_key)

    # test_input_1 = "Hello"
    # print(get_response(client, test_input_1))
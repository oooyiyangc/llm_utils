"""
This is an example to demonstrate the use of this util package.
The example will take a list of OCR texts and clean them using the OpenAI API.
"""

import time
from llm_api import query_openai
from llm_api import query_claude


def get_response_safe(client, user_input, system_prompt=None, model="gpt-4o-mini", max_tokens=1024, temperature=0):
    """
    This function is a wrapper around query_*.get_response() that retries the query up to 5 times if it encounters an error.

    Args:
        client (openai.Client): OpenAI API client
        user_input (str): User input text
        system_prompt (str): System prompt text
        model (str): Model name
        max_tokens (int): Maximum tokens
        temperature (float): Temperature

    Returns:
        str: Response text
    """

    if user_input == "" or user_input.strip() == "":
        return ""
    for num_retry in range(5):
        try:
            return query_openai.get_response(client, user_input, system_prompt=system_prompt, model=model, max_tokens=max_tokens, temperature=temperature)
        except Exception as e: # handles RateLimit or Timeout Error
            print(f"Encountered error: {repr(e)}. Will retry after sleeping for {15 * (2 ** (num_retry))} seconds (attempt {num_retry+1}/5)")
            time.sleep(15 * (2 ** (num_retry)))


if __name__ == "__main__":

    system_prompt = "This is a news article from a historical newspaper. It has been run through OCR software, but there are some errors. Could you correct them? You can remove line breaks ('\n'), but please do not remove paragraph breaks ('\n\n'). You can merge words that are split between two lines. Do not add any comments."
    
    client = query_openai.create_client(api_key="sk-...")

    # import a list of texts that you want to run through the LLM (can be a list, a dict, a pandas dataframe, etc.)
    my_list_of_ocr_texts = [
        "The quick bron fox\njumps over the lazy dog.",
        "The quick brown foxjumps over the lazy dog.",
        "The quick brown fox\njumps ower the lazy dog.",
        "The quiek brown fox\njumps over the la\nzy dog.",
        "The quick brown fox\njumps over the lazzzzy dog.",
    ]

    # iterate over the list and get_response for each input
    cleaned_texts = []
    for ocr_text in my_list_of_ocr_texts:
        response = get_response_safe(client, ocr_text, system_prompt=system_prompt, model="gpt-4o-mini", max_tokens=1024, temperature=0)
        cleaned_text = response.strip()
        cleaned_texts.append(cleaned_text)
        print(f"Original OCR text: {ocr_text}")
        print(f"Cleaned text: {cleaned_text}")
        print()

from openai import OpenAI
import os
import warnings
import json

from tqdm import tqdm
from typing import List, Tuple, Optional, Union

OPENAI_MODEL_ALIAS = {
    "gpt-3.5": "gpt-3.5-turbo-0125", 
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini"
}
MAX_TOKENS = 16384

def prepare_batch_request(
        client: OpenAI, 
        job_name: str, 
        user_inputs: List[str], 
        system_prompts: List[str], 
        ids: Optional[List[Union[int, str]]], 
        model: str = "gpt-4o-mini", 
        response_schema: Optional[dict] = None,
        max_tokens: int = 1024, 
        temperature: float = 0, 
        top_p: float = 1
) -> str:
    """
    Create batch request for OpenAI API
    
    Args:
        client (OpenAI): OpenAI client
        job_name (str): Job name
        user_inputs (List[str]): List of user inputs
        system_prompts (List[str]): List of system prompts
        ids (List[Union[int, str]]): List of IDs
        model (str): Model to use
        max_tokens (int): Maximum number of tokens
        temperature (float): Temperature for sampling
        
    Returns:
        List[str]: List of responses from OpenAI API
    """

    if model in OPENAI_MODEL_ALIAS:
        model = OPENAI_MODEL_ALIAS[model]
    
    # print(f"Creating batch request with model {model}...")

    # sanity checks
    # make sure user_inputs, system_prompts and ids (if given) have the same length
    if len(user_inputs) != len(system_prompts):
        raise ValueError(f"Length of user_inputs and system_prompts should be the same, but got {len(user_inputs)} and {len(system_prompts)}")

    if ids is not None and len(user_inputs) != len(ids):
        raise ValueError(f"Length of user_inputs and ids should be the same, but got {len(user_inputs)} and {len(ids)}")


    batch_request = []
    default_id = 0
    for user_input, system_prompt in tqdm(zip(user_inputs, system_prompts)):

        if ids is not None:
            id = ids[default_id]
        else:
            id = default_id

        if system_prompt is None:
            warnings.warn("No system prompt provided. Using default system prompt.")
            batch_request.append(
                {
                    "custom_id": f"{job_name}_{id}",
                    "method": "POST", 
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ],
                        "temperature": temperature,
                        "max_completion_tokens": min(max_tokens, MAX_TOKENS),
                        "top_p": 1, 
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "seed": 42
                    }
                }
            )
        else:
            batch_request.append(
                {
                    "custom_id": f"{job_name}_{id}",
                    "method": "POST", 
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            }, 
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ],
                        "temperature": temperature,
                        "max_completion_tokens": min(max_tokens, MAX_TOKENS),
                        "top_p": top_p,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "seed": 42
                    }
                }
            )
        

        if response_schema is not None:
            batch_request[-1]["body"]["response_format"] = response_schema

        default_id += 1

    # save batch request to JSONL file
    with open(f"{job_name}.jsonl", "w") as f:
        for item in batch_request:
            f.write(json.dumps(item) + "\n")
    
    # upload batch request to OpenAI API
    batch_input_file = client.files.create(
        file=open(f"{job_name}.jsonl", "rb"),
        purpose="batch"
    )

    print(f"Batch request uploaded with ID {batch_input_file.id}")
    print("Details:")
    print(batch_input_file)
    
    return batch_input_file.id


def create_batch_request(client: OpenAI, batch_input_file_id: str, job_name: str, description: Optional[str] = None) -> str:
    """
    Create batch request for OpenAI API
    
    Args:
        client (OpenAI): OpenAI client
        batch_input_file_id (str): Batch input file ID
        job_name (str): Job name
        description (str): Description of the job
        
    Returns:
        str: Batch request ID
    """

    # create batch request
    batch_request = client.batches.create(
        input_file_id=batch_input_file_id, 
        endpoint="/v1/chat/completions", 
        completion_window="24h", # this cannot be changed
        metadata = {
            "description": job_name if description is None else f"{job_name}: {description}"
        }
    )

    print(f"Batch request created with ID {batch_request.id}")
    print("Details:")
    print(batch_request)

    # save batch request to JSON file
    batch_request_to_save = dict(batch_request).copy()
    batch_request_to_save["request_counts"] = dict(batch_request.request_counts)
    batch_request_to_save["__exclude_fields__"] = dict(batch_request.metadata)
    with open(f"{job_name}_batch_request.json", "w") as f:
        json.dump(dict(batch_request_to_save), f, indent=4)

    return batch_request.id


def check_batch_request(client: OpenAI, batch_request_id: str, job_name: str) -> str:
    """
    Check batch request status

    Args:
        client (OpenAI): OpenAI client
        batch_request_id (str): Batch request ID
        job_name (str): Job name

    Returns:
        str: Batch request ID
    """

    # check batch request
    batch_request = client.batches.retrieve(batch_request_id)

    print(f"Checking batch request with ID {batch_request_id}")
    print(f"Current status: {batch_request.status}")
    print("Details:")
    print(batch_request)

    return batch_request.id


def retrieve_results(client: OpenAI, batch_request_id: str, job_name: str, output_file_dir: Optional[str] = None) -> str:
    """
    Retrieve results from batch request

    Args:
        client (OpenAI): OpenAI client
        batch_request_id (str): Batch request ID
        job_name (str): Job name
        output_file_dir (str): Output file directory

    Returns:
        str: Output file local path
    """
    
    # check batch request
    batch_request = client.batches.retrieve(batch_request_id)

    if batch_request.status != "completed":
        print(f"Batch request with ID {batch_request_id} is not completed yet. Current status: {batch_request.status}")
    else:
        print(f"Batch request with ID {batch_request_id} is completed. ")
        print(f"Output file ID: {batch_request.output_file_id}")
    
        # retrieve successful results
        file_response = client.files.content(batch_request.output_file_id).text

        # save successful results to JSONL file
        if output_file_dir is not None:
            output_file_path = os.path.join(output_file_dir, f"{job_name}_results.jsonl")
        else:
            output_file_path = f"{job_name}_results.jsonl"
        with open(output_file_path, "w") as f:
            f.write(file_response)
        
        # retrieve error file
        if batch_request.error_file_id is not None:
            print(f"Batch request with ID {batch_request_id} has errors. ")
            failed_results = client.files.content(batch_request.error_file_id).text

            # save failed results to JSONL file
            if output_file_dir is not None:
                error_file_path = os.path.join(output_file_dir, f"{job_name}_errors.jsonl")
            else:
                error_file_path = f"{job_name}_errors.jsonl"
            with open(error_file_path, "w") as f:
                f.write(failed_results)
    
    return output_file_path



if __name__ == "__main__":

    OPENAI_API_KEY = "sk-..."
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # print("Creating client...")
    # client = create_client()

    # test_input_1 = "Hello"
    # print(get_response(client, test_input_1))
"""
This is an example to demonstrate how to use the batch request API to send multiple requests to the OpenAI API at once. This also 
illustrates how to use structured output. 

In this example, we want to use GPT to extract multiple attributes from a list of editorials about college protests from the 1960s. 
The main pipeline is as follows:
1. load data and create mini-batches
2. send mini-batches to OpenAI API
3. retrieve results
4. post-process results

Important notes:
- You will want to test your code first before creating jobs
- Make sure you specify the ids (in create_mini_batches function) correctly so you can track the results (the output may not be in the same order as the input)
"""

import argparse
import json
import os

from typing import List, Optional, Tuple, Union

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

from llm_api import batch_request_openai, query_openai


def create_mini_batches(inputs: List[str], system_prompts: Union[str, List[str]], batch_size: int = 1000, starting_batch_idx: Optional[int] = 0) -> Tuple[List[str], List[Union[int, str]], List[str]]:
    """
    Create mini-batches of inputs and their corresponding system prompts.

    Args:
        inputs (List[str]): The list of input texts to process.
        system_prompts (Union[str, List[str]]): The system prompts to use for each input. Can be a single prompt or a list of prompts.
        batch_size (int, optional): The size of each mini-batch. Defaults to 1000.
        starting_batch_idx (int, optional): The index of the batch to start from. Defaults to 0.

    Returns:
        A tuple containing the mini-batch of articles, their IDs, and the corresponding system prompts.
    """

    if isinstance(system_prompts, str):
        system_prompts = [system_prompts] * len(inputs)
        print("Provided single system prompt. Using it for all inputs.")
    else:
        if len(system_prompts) != len(inputs):
            raise ValueError(f"Expected prompts to match inputs in length, but got {len(system_prompts)} prompts and {len(inputs)} inputs.")

    mini_batch = inputs[starting_batch_idx*batch_size:(starting_batch_idx+1)*batch_size]

    # ---- CUSTOMIZE THIS: preprocessing inputs -------------------------

    headlines = [art["data"]["headline"] for art in mini_batch]
    bylines = [art["data"]["byline"] for art in mini_batch]
    texts = [art["data"]["text"] for art in mini_batch]

    articles = []
    system_prompts = []
    for i in range(len(mini_batch)):
        curr_art = " ".join([arg for arg in [headlines[i], bylines[i], texts[i]] if isinstance(arg, str) and len(arg) > 0])
        articles.append(curr_art)
        system_prompts.append(system_prompts[i])

    # ids is a list of article ids
    ids = [art["id"] for art in mini_batch]

    # -------------------------------------------------------------------

    return articles, ids, system_prompts


def create_jobs(client: OpenAI, inputs: List[str], system_prompts: List[str], task_name: str, task_description: str, working_dir: str, batch_size: int = 1000, response_schema: Optional[dict] = None):
    """
    Create jobs for processing articles in batches.

    Args:
        client (OpenAI): The OpenAI client.
        inputs (List[str]): The list of article inputs.
        system_prompts (List[str]): The list of system prompts.
        task_name (str): The name of the task.
        task_description (str): The description of the task.
        working_dir (str): The working directory for saving outputs.
        batch_size (int): The size of each batch.
        response_schema (Optional[dict]): The schema for the response.

    Returns:
        None
    """

    # get confirmation from user
    print(f"Number of articles to run inference on: {len(inputs)}")
    print(f"Number of batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    confirm = input("Type 'yes' to proceed: ")
    if confirm != "yes":
        print("Exiting...")
        exit()

    batch_request_metadata = {}

    for batch_id in range(num_batches):
    
        # create one mini batch
        articles, ids, system_prompts = create_mini_batches(inputs, system_prompts, batch_size, batch_id)

        job_name = f"{task_name}_minibatch_{batch_id}"

        print("Preparing batch request...")
        batch_file_id = batch_request_openai.prepare_batch_request(client, job_name, articles, system_prompts, ids, model="gpt-4o-mini", response_schema=response_schema, max_tokens=1024, temperature=1, top_p=0)

        print("Creating batch request...")
        batch_request_id = batch_request_openai.create_batch_request(client, batch_file_id, job_name, f"{task_description}, mini-batch {batch_id}")

        batch_request_metadata[batch_id] = {
            "batch_request_id": batch_request_id,
            "batch_file_id": batch_file_id,
            "job_name": job_name
        }

    print(f"Created {num_batches} batch requests.")

    # save batch request metadata
    os.makedirs(os.path.join(working_dir, "batch_request_metadata", task_name), exist_ok=True)
    with open(os.path.join(working_dir, "batch_request_metadata", task_name, "batch_request_metadata.json"), "w") as f:
        json.dump(batch_request_metadata, f, indent=4)

    print(f"✅ Jobs created; Batch request metadata saved to {os.path.join(working_dir, 'batch_request_metadata', task_name, 'batch_request_metadata.json')}")


def retrieve_results(working_dir, task_name, num_batches):
    """
    Retrieve results from batch requests.

    Args:
        working_dir (str): The working directory.
        task_name (str): The name of the task.
        num_batches (int): The number of batches.

    Returns:
        List[str]: A list of paths to the result files.
    """

    # load batch request metadata 
    with open(os.path.join(working_dir, "batch_request_metadata", task_name, "batch_request_metadata.json")) as f:
        batch_request_metadata = json.load(f)

    all_results_path = []

    for batch_id in range(num_batches):

        batch_request_id = batch_request_metadata[str(batch_id)]["batch_request_id"]
        job_name = batch_request_metadata[str(batch_id)]["job_name"]

        print("Checking batch request status...")
        batch_request_openai.check_batch_request(client, batch_request_id, job_name)

        print("Retrieving batch request results...")
        os.makedirs(os.path.join(working_dir, "batched_output", task_name), exist_ok=True)
        results_path = batch_request_openai.retrieve_results(client, batch_request_id, job_name, os.path.join(working_dir, "batched_output", task_name))
        all_results_path.append(results_path)
    
    print("✅ All results retrieved.")
    
    return all_results_path


def post_process(working_dir, task_name, num_batches):
    """
    Post-process the results from batch requests.

    Args:
        working_dir (str): The working directory.
        task_name (str): The name of the task.
        num_batches (int): The number of batches.

    Returns:
        None
    """

    results = []
    structured_results = []

    print("Merging results from mini-batches...")
    for i in tqdm(range(num_batches)):
        with open(os.path.join(working_dir, "batched_output", task_name, f"{task_name}_minibatch_{i}_results.jsonl"), 'r') as json_file:
            json_list = list(json_file)

        for line in json_list:
            result = json.loads(line)
            results.append(result)

            # article_id = "_".join(result['custom_id'].split("_")[6:])
            job_name = f"{task_name}_minibatch_{i}"
            article_id = result['custom_id'].replace(job_name + "_", "")

            response = result['response']['body']['choices'][0]['message']['content']
            try:
                openai_responses = json.loads(response)
                structured_results.append({
                    "id": article_id,
                    **openai_responses
                })
            except json.JSONDecodeError:
                # Likely GPT output exceeded token limit and JSON got truncated
                print(f"JSONDecodeError: could not parse response for article {article_id}")
                print(response)
                continue
            except Exception as e:
                print(f"Unexpected error parsing response for article {article_id}: {e}")

    os.makedirs(os.path.join(working_dir, "output"), exist_ok=True)
    with open(os.path.join(working_dir, "output", f"{task_name}_batched_results_raw.json"), 'w') as f:
        json.dump(results, f, indent=4)

    structured_results_df = pd.DataFrame(structured_results)
    # structured_results_df["year"] = structured_results_df["id"].apply(lambda x: x.split("-")[-3])

    # save csv
    structured_results_df.to_csv(os.path.join(working_dir, "output", f"{task_name}_batched_results_structured.csv"), index=False)

    print(f"✅ Final results saved as {task_name}_batched_results_structured.json and .csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process batch requests for editorial analysis")
    parser.add_argument("--step", "-s", type=str, choices=["create_jobs", "retrieve_results", "post_process"], help="Stage of the pipeline to run")
    args = parser.parse_args()

    # ---- CUSTOMIZE THIS: metadata settings -------------------------

    task_name = "college_protests"
    task_description = "Stance extraction from college protest editorials"
    working_dir = f"/mnt/data01/college_protests/editorials/gpt_pipeline/inference/"
    secrets_file = "secrets.yaml"
    system_prompt_file = "prompt.txt"
    batch_size = 3000

    # structured output schema
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "protest_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "protest_reason": { "type": "string" },
                    "is_protestor_violent": { "type": "boolean" },
                    "authority_response": { "type": "string" },
                    "is_authority_violent": { "type": "boolean" },
                    "stance": { "type": "string" },
                    "stance_sentence": { "type": "string" },
                    "is_sympathetic_to_students": { "type": "boolean" }
                },
                "required": [
                    "protest_reason",
                    "is_protestor_violent",
                    "authority_response",
                    "is_authority_violent",
                    "stance",
                    "stance_sentence", 
                    "is_sympathetic_to_students"
                ],
                "additionalProperties": False
            }
        }
    }

    # -------------------------------------------------------------

    

    # ---- CUSTOMIZE THIS: loading inputs -------------------------

    inputs = []

    years = range(1965, 1970+1)

    for year in years:
        with open(f"/mnt/data01/college_protests/editorials/results/{year}/college_protests_editorials_{year}.json", "r") as f:
            editorials = json.load(f)
        inputs.extend(editorials)

    # -------------------------------------------------------------

    # check inputs is an iterable
    if not isinstance(inputs, list):
        raise ValueError("Inputs should be a list")
    print(f"Length of inputs: {len(inputs)}")

    with open(secrets_file, "r", encoding="utf-8") as f:
        secrets = yaml.safe_load(f)
    api_key = secrets.get("OPENAI_API_KEY")

    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # create batch request
    client = query_openai.create_client(api_key)

    # create mini batches
    num_batches = (len(inputs) + batch_size - 1) // batch_size

    # select a step
    if args.step:
        step = args.step
    else:
        step = "create_jobs"
    assert step in ["create_jobs", "retrieve_results", "post_process"]
    
    # run pipeline
    if step == "create_jobs":
        create_jobs(client, inputs, system_prompt, task_name, task_description, working_dir, batch_size, response_schema=response_schema)
    
    elif step == "retrieve_results":
        retrieve_results(working_dir, task_name,num_batches)

    elif step == "post_process":
        post_process(working_dir, task_name, num_batches)


    
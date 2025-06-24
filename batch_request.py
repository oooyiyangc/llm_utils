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

import json
import pandas as pd
import os
from tqdm import tqdm

from llm_api import query_openai, batch_request_openai

def create_mini_batches(inputs, batch_size=1000, starting_batch_idx=0):

    mini_batch = inputs[starting_batch_idx*batch_size:(starting_batch_idx+1)*batch_size]

    headlines = [art["data"]["headline"] for art in mini_batch]
    bylines = [art["data"]["byline"] for art in mini_batch]
    texts = [art["data"]["text"] for art in mini_batch]

    articles = []
    system_prompts = []
    for i in range(len(mini_batch)):
        curr_art = " ".join([arg for arg in [headlines[i], bylines[i], texts[i]] if isinstance(arg, str) and len(arg) > 0])
        articles.append(curr_art)

        system_prompt = """I will give you an editorial about student protest from the 1960s. Can you help me identify the following:
- What were students protesting about?
- Did the protestors use violence (e.g. fight, arson, shooting)
- What were the responses from the police, schools, or local authorities?
- Did the police, schools, or local authorities use violence against protestors? 
- What is the stance or opinions on this protest in the editorial?
- Which sentence in the editorial best summarizes the editorial's stance or opinions?
- Are they sympathetic with students (or side with students) or not?
"""

        system_prompts.append(system_prompt)

    # ids is a list of article ids
    ids = [art["id"] for art in mini_batch]

    return articles, ids, system_prompts


def create_jobs(client, inputs, batch_size=1000, response_schema=None):

    num_batches = (len(inputs) + batch_size - 1) // batch_size

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
        articles, ids, system_prompts = create_mini_batches(inputs, batch_size, batch_id)

        job_name = f"college_protests_minibatch_{batch_id}"

        print("Preparing batch request...")
        batch_file_id = batch_request_openai.prepare_batch_request(client, job_name, articles, system_prompts, ids, model="gpt-4o-mini", response_schema=response_schema, max_tokens=1024, temperature=1, top_p=0)

        print("Creating batch request...")
        batch_request_id = batch_request_openai.create_batch_request(client, batch_file_id, job_name, f"Stance extraction from college protest editorials, mini-batch {batch_id}")

        batch_request_metadata[batch_id] = {
            "batch_request_id": batch_request_id,
            "batch_file_id": batch_file_id,
            "job_name": job_name
        }

    print(f"Created {num_batches} batch requests.")

    # save batch request metadata
    os.makedirs(f"/mnt/data01/college_protests/editorials/gpt_pipeline/inference/batch_request_metadata/", exist_ok=True)
    with open(f"/mnt/data01/college_protests/editorials/gpt_pipeline/inference/batch_request_metadata/batch_request_metadata.json", "w") as f:
        json.dump(batch_request_metadata, f, indent=4)
    
    print("Batch request metadata saved to batch_request_metadata/batch_request_metadata.json")


def retrieve_results(num_batches):

    # load batch request metadata
    with open(f"/mnt/data01/college_protests/editorials/gpt_pipeline/inference/batch_request_metadata/batch_request_metadata.json") as f:
        batch_request_metadata = json.load(f)

    for batch_id in range(num_batches):

        batch_request_id = batch_request_metadata[str(batch_id)]["batch_request_id"]
        job_name = batch_request_metadata[str(batch_id)]["job_name"]

        print("Checking batch request status...")
        batch_request_openai.check_batch_request(client, batch_request_id, job_name)

        print("Retrieving batch request results...")
        os.makedirs(f"/mnt/data01/college_protests/editorials/gpt_pipeline/inference/batched_output/", exist_ok=True)
        results_path = batch_request_openai.retrieve_results(client, batch_request_id, job_name, f"/mnt/data01/college_protests/editorials/gpt_pipeline/inference/batched_output/")


def post_process(num_batches):

    results = []
    structured_results = []

    print("Merging results from mini-batches...")
    for i in tqdm(range(num_batches)):
        with open(f'/mnt/data01/college_protests/editorials/gpt_pipeline/inference/batched_output/college_protests_minibatch_{i}_results.jsonl', 'r') as json_file:
            json_list = list(json_file)

        for line in json_list:
            result = json.loads(line)
            results.append(result)

            article_id = "_".join(result['custom_id'].split("_")[6:])
            response = result['response']['body']['choices'][0]['message']['content']
            try:
                openai_responses = json.loads(response)
                structured_results.append({
                    "id": article_id,
                })
                for key in openai_responses:
                    structured_results[-1][key] = openai_responses[key]
            except:
                # likely GPT output encounter max token limit, so the json output got cut off
                print(f"Error parsing response for article {article_id}")
                print(response)
    
    os.makedirs(f'/mnt/data01/college_protests/editorials/gpt_pipeline/inference/output/', exist_ok=True)
    with open(f'/mnt/data01/college_protests/editorials/gpt_pipeline/inference/output/college_protests_editorials_batched_results_raw.json', 'w') as f:
        json.dump(results, f, indent=4)

    structured_results_df = pd.DataFrame(structured_results)
    structured_results_df["year"] = structured_results_df["id"].apply(lambda x: x.split("-")[-3])

    # save csv
    structured_results_df.to_csv(f'/mnt/data01/college_protests/editorials/gpt_pipeline/inference/output/college_protests_editorials_structured_results.csv', index=False)

    

if __name__ == "__main__":

    years = range(1965, 1970+1)

    all_editorials = []

    for year in years:
            
        with open(f"/mnt/data01/college_protests/editorials/results/{year}/college_protests_editorials_{year}.json", "r") as f:
            editorials = json.load(f)

        all_editorials.extend(editorials)
    
    print(len(all_editorials))


    # define structured output schema
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

    # create batch request
    client = query_openai.create_client("sk-...")


    # create mini batches
    batch_size = 3000

    # select a step
    step = "create_jobs"
    assert step in ["create_jobs", "retrieve_results", "post_process"]
    
    
    # run pipeline
    if step == "create_jobs":
        create_jobs(all_editorials, batch_size, response_schema=response_schema)
    
    elif step == "retrieve_results":
        retrieve_results(num_batches)

    elif step == "post_process":
        post_process(num_batches)
    
    
    
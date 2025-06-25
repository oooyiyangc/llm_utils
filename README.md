# llm_utils

A utility package for running and managing large language model (LLM) requests, with a focus on batch processing, structured output, and multi-model support (OpenAI, Anthropic Claude, and Llama 2). The project provides reusable modules and example scripts for cleaning OCR text, extracting structured data, and running LLM inference at scale.

## Features
- **Single and Batch LLM Requests**: Utilities for sending individual or batched requests to LLM APIs.
- **Structured Output**: Support for extracting structured data (JSON schema) from LLM responses.
- **Multi-Model Support**: Integrations for OpenAI (GPT-3.5, GPT-4o), Anthropic Claude, and Llama 2.
- **Example Pipelines**: Scripts for cleaning OCR text, extracting information from historical documents, and analyzing editorials.

## Project Structure
```
llm_utils/
  batch_request.py                # Example: batch extraction from editorials (OpenAI)
  extract_item.py                 # Example: batch extraction from OCR book pages (OpenAI)
  main.py                         # Example: single-request OCR text cleaning (OpenAI)
  llm_api/
    batch_request_openai.py       # Batch request utilities for OpenAI API
    query_claude.py               # Anthropic Claude API utilities
    query_llama.py                # Llama 2 API utilities
    query_openai.py               # OpenAI API utilities
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd llm_utils
   ```
2. **Install dependencies:**
   ```bash
   pip install openai anthropic tqdm pandas pyyaml
   ```
   (You may also need to install Llama 2 and its dependencies for `query_llama.py`.)

## API Keys & Configuration
- **OpenAI:** Set your API key in `secrets.yaml` or as the `OPENAI_API_KEY` environment variable.
- **Anthropic Claude:** Set your API key as the `ANTHROPIC_API_KEY` environment variable.
- **Llama 2:** Requires local model files and configuration (see `llm_api/query_llama.py`).

## Example Usage

### 1. Clean OCR Text with OpenAI (Single Request)
Run `main.py` to clean a list of OCR texts using the OpenAI API:
```bash
python main.py
```
- Edits and corrects OCR errors in historical newspaper articles.
- Requires an OpenAI API key.

### 2. Batch Extraction from Editorials (OpenAI)
Run `batch_request.py` to extract structured information from a batch of editorials:
```bash
python batch_request.py
```
- Loads data, creates mini-batches, sends batch requests, and post-processes results.
- Example schema: protest reasons, violence, authority response, stance, etc.
- Requires OpenAI API key and input data in the expected format.

### 3. Batch Extraction from OCR Book Pages
Run `extract_item.py` to extract structured scientific/technical innovations from OCR'd book pages:
```bash
python extract_item.py
```
- Loads OCR text files from `data/Tech_History_Book_txt/`.
- Uses a system prompt from `prompt.txt` and API key from `secrets.yaml`.
- Outputs structured JSON with extracted records.

## Modules
- `llm_api/query_openai.py`: Utilities for single OpenAI requests.
- `llm_api/batch_request_openai.py`: Batch request creation, status checking, and result retrieval for OpenAI.
- `llm_api/query_claude.py`: Anthropic Claude API utilities.
- `llm_api/query_llama.py`: Llama 2 API utilities (requires local model files).

## Requirements
- Python 3.8+
- openai==1.61.1
- anthropic==0.34.2
- tqdm
- pandas
- pyyaml

## Notes
- For batch jobs, ensure you have the correct input data and update file paths as needed.
- For Llama 2, you must provide your own model checkpoints and tokenizer.
- Example scripts are meant as templates and may require adaptation for your data and workflow.

## License
- Llama 2 integration is subject to the [Llama 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
- Other code is MIT licensed unless otherwise specified. 

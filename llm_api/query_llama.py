# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Usage: CUDA_VISIBLE_DEVICES=X torchrun --nproc_per_node=1 run_inference.py

from typing import List, Optional
from llama import Llama, Dialog
import warnings

def build(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 8
):
    """
    Build a Llama generator object.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.

    Returns:
        Llama: A Llama generator object.
    """

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    return generator

def create_client(ckpt_dir="llama-2-7b-chat/", tokenizer_path="tokenizer.model"):
    """
    Create a Llama client object.

    Returns:
        Llama: A Llama client object.
    """

    client = build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
    )

    return client


def generate_output(
    user_input: str,
    generator: Llama,
    system_prompt: Optional[str] = None,
    history: List[Dialog] = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        user_input (str): The user input prompt for generating text.
        generator (Llama): A Llama generator object.
        system_prompt (str, optional): The system prompt for generating text. If None, the system
            prompt will be set to the last generated response. Defaults to None.
        history (List[Dialog], optional): The chat history to be used for generating text.
            If None, a new chat history will be created. Defaults to None.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    if history is None:
        if system_prompt is not None:
            dialogs: List[Dialog] = [
                [{"role": "system", "content": system_prompt}, 
                 {"role": "user", "content": user_input}]
            ]
        else:
            dialogs: List[Dialog] = [
                [{"role": "user", "content": user_input}]
            ]
    else:
        dialogs = history
        dialogs[-1].append({"role": "user", "content": user_input})

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content'].strip()}"
        )
        print("\n==================================\n")

    # append the generated response to the last dialog
    dialogs[-1].append(result["generation"])

    return dialogs

def get_response(client, user_input, system_prompt=None, model=None, max_tokens=1024, temperature=0):

    if model is not None:
        warnings.warn("model argument is not used. The model is determined by the client object.")
    
    dialogs = generate_output(user_input, client, system_prompt=system_prompt, temperature=temperature, max_gen_len=max_tokens)
    if dialogs[-1][-1]["role"] == "assistant":
        return dialogs[-1][-1]["content"]


if __name__ == "__main__":

    print("Loading model...")
    generator = build(
        ckpt_dir="llama-2-7b-chat/",
        tokenizer_path="tokenizer.model",
    )
    print()
    print("Model loaded successfully! ")
    print("Enter your prompt after the '>>>' symbol. Type '//exit' or '//quit' to end the session.")

    dialogs = None
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ["//exit", "//quit", "//q"]:
            print("Goodbye!")
            break
        dialogs = generate_output(user_input, generator, history=dialogs)

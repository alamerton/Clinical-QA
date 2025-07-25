import sys
import os
from datetime import datetime
import tiktoken
import random
import json
import re
import ast
import subprocess

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.generation.call_mimic_iii import call_mimic_iii

NUMBER_OF_QA_SETS = 1000
MODEL_NAME = "gpt-4"


def save_dataset(dataset, directory: str, num_qa_sets):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = f"""data/{directory}/{num_qa_sets}-QA-sets-{date}"""
    with open(f"{output_path}.json", "w") as json_file:
        json.dump(dataset, json_file, indent=4)


def save_checkpoint(
    dataset, row, checkpoint_interval, directory="data/generations/checkpoints/"
):
    if (row + 1) % checkpoint_interval == 0:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checkpoint_name = f"{row+1}-rows-{date}"
        checkpoint_path = directory + checkpoint_name
        with open(f"{checkpoint_path}.json", "w") as json_file:
            json.dump(dataset, json_file, indent=4)


def count_tokens(text, model):
    if "gpt-4o" in model:
        encoder = tiktoken.get_encoding("o200k_base")  # GPT-4o
    else:
        encoder = tiktoken.encoding_for_model(model)  # GPT-3
    return len(encoder.encode(text))


def calculate_average_tokens(strings, model_name=MODEL_NAME):
    total_tokens = 0
    for string in strings:
        if type(string) == tuple:
            string = str(string)
        total_tokens += count_tokens(string, model_name)
    average_tokens = total_tokens / len(strings)
    return average_tokens


def calculate_max_tokens(strings, model_name=MODEL_NAME):
    max_tokens = 0
    for string in strings:
        if type(string) == tuple:
            string = str(string)
        token_count = count_tokens(string, model_name)
        if token_count > max_tokens:
            max_tokens = token_count
    return max_tokens


def calculate_max_discharge_summaries(model_name, limit=10):
    # return the maximum number of discharge summaries you can send
    # a model
    biggest_ds_strings = []
    for i in range(0, limit):
        strings = call_mimic_iii(NUMBER_OF_QA_SETS, i)
        biggest_ds_strings.append(calculate_max_tokens(strings, model_name))
    return biggest_ds_strings


def select_capability_type(factual_proportion: int, reasoning_proportion: int) -> str:
    # Calculate the total weight to allow arbitrary proportions
    total_weight = factual_proportion + reasoning_proportion

    if total_weight == 0:
        raise ValueError("Total weight must be greater than zero.")

    # Generate a random number between 0 and the total weight
    random_value = random.randrange(total_weight)

    # Determine the selection based on the random value
    return "Factual QA" if random_value < factual_proportion else "Reasoning QA"


def parse_llm_segments(segment_string):
    """
    Parses LLM output expected to be a list of dictionaries:
    [{"name": ..., "text": ...}, ...]
    Returns the parsed list.
    """
    segment_string = segment_string.strip()
    segment_string = re.sub(r"^```(?:json)?\s*", "", segment_string)
    segment_string = re.sub(r"\s*```$", "", segment_string).strip()
    segment_string = (
        segment_string.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    # Try JSON
    try:
        return json.loads(segment_string)
    except json.JSONDecodeError:
        pass

    # Try ast.literal_eval
    try:
        result = ast.literal_eval(segment_string)
        if isinstance(result, list) and all(
            isinstance(x, dict) and "name" in x and "text" in x for x in result
        ):
            return result
    except Exception:
        pass

    raise ValueError("Failed to parse LLM segment output")


def filter_segments(segments, min_sentences=1, min_words=10):
    filtered = []
    for seg in segments:
        text = seg.get("text", "").strip()
        sentences = re.split(r"[.!?]\s+", text)
        word_count = len(text.split())
        if len(sentences) >= min_sentences and word_count >= min_words:
            filtered.append(seg)
    return filtered


def turn_tunnelblick_on():
    print("Switching VPN on...")
    subprocess.run(
        [
            "osascript",
            "-e",
            f'tell application "Tunnelblick" to connect "{os.getenv("TUNNELBLICK_CONFIG")}"',
        ]
    )


def turn_tunnelblick_off():
    print("Switching VPN off...")
    subprocess.run(
        [
            "osascript",
            "-e",
            f'tell application "Tunnelblick" to disconnect "{os.getenv("TUNNELBLICK_CONFIG")}"',
        ]
    )

# Given a note or multiple notes
# Pass notes to GPT as context
# Prompt model to generate a question and answer based on the context
# with specifications and requirements for the type of question

import os
import random
import time
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.core.exceptions import HttpResponseError
from utils.generation.prompts import (
    get_factual_generation_prompt,
    get_reasoning_generation_prompt,
    get_segmentation_prompt,
)

load_dotenv()


def get_question_type():
    question_types = [
        "Knowledge and Recall: Questions focused on factual information retrieval. For example: 'What is the normal range of blood pressure?'",
        "Comprehension: Questions requiring conceptual understanding or interpretation. For example: 'Why is hypertension a risk factor for stroke?'",
        "Application and Analysis: Questions involving clinical problem-solving or scenario-based reasoning. For example: 'What is the best initial treatment for a patient presenting with chest pain?'",
        "Synthesis and Evaluation: Questions requiring complex decision-making and comparison. For example: 'Compare the risks of drug A and drug B for treating atrial fibrillation in elderly patients.'",
    ]
    return random.choice(question_types)


def call_gpt(model_name, discharge_summary_string, capability_type):

    max_retries = 10
    retry_delay = 5
    question_type = get_question_type()

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    if capability_type == "Factual QA":
        system_message, user_prompt = get_factual_generation_prompt(
            question_type, discharge_summary_string
        )
    elif capability_type == "Reasoning QA":
        system_message, user_prompt = get_reasoning_generation_prompt(
            discharge_summary_string
        )
    else:
        raise ValueError("Invalid capability type passed to call_gpt")

    for i in range(0, max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=999,
                temperature=1,
            )
            return response.choices[0].message.content

        except HttpResponseError as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Attempt {i + 1} of {max_retries}.")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
        raise RuntimeError("Maximum retries exceeded.")


def call_llm_for_segmentation(model_name, discharge_summary_string, capability_type):

    max_retries = 10
    retry_delay = 5
    question_type = get_question_type()

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    if capability_type == "Factual QA":
        system_message, user_prompt = get_factual_generation_prompt(
            question_type, discharge_summary_string
        )
    elif capability_type == "Reasoning QA":
        system_message, user_prompt = get_segmentation_prompt(discharge_summary_string)
    else:
        raise ValueError("Invalid capability type passed to call_gpt")

    for i in range(0, max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=999,
                temperature=1,
            )
            return response.choices[0].message.content

        except HttpResponseError as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Attempt {i + 1} of {max_retries}.")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
        raise RuntimeError("Maximum retries exceeded.")

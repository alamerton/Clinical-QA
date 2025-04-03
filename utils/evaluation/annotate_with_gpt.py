import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

ANNOTATION_MODEL = "gpt-4o"


def annotate_with_gpt(
    discharge_summary,
    question,
    expected_answer,
):

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_GPT_4O_ENDPOINT"),
        api_key=os.getenv("AZURE_GPT_4O_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    system_message = """You are an expert medical professional and educational assessment specialist tasked 
    with analyzing questions from clinical cases according to Bloom's Taxonomy of cognitive levels."""

    user_prompt = f"""
        Your task is to classify the provided clinical question according to Bloom's Taxonomy of cognitive levels.
        Analyze both the question and its corresponding answer to determine the highest cognitive level being tested.

        Classification Instructions:
        1. Remember: Questions testing recall of medical facts, terminology, or basic concepts
        2. Understand: Questions requiring comprehension and explanation of medical concepts
        3. Apply: Questions asking to use information in new situations or implement knowledge
        4. Analyze: Questions requiring breaking information into parts and understanding relationships
        5. Evaluate: Questions asking to justify decisions or make judgments based on criteria
        6. Create: Questions requiring synthesis of information to create new patterns or solutions

        For each question, provide:
        1. The primary Bloom's level (1-6)
        2. A brief (1-2 sentence) justification for the classification
        3. Key cognitive skills being tested

        Output format:
        Bloom's Level: [1-6]
        Justification: [Your explanation]
        Cognitive Skills: [List of key skills tested]

        Discharge Summary:
        {discharge_summary}

        Question: {question}

        Answer: {expected_answer}
    """

    response = client.chat.completions.create(
        model=ANNOTATION_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=10,
        temperature=0,
    )

    return response.choices[0].message.content

import pandas as pd
from tqdm import tqdm
import sys
import os
import numpy as np
from rouge import Rouge
import numpy as np
from nltk.util import ngrams
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize

# from deepeval.metrics import GEval
# from deepeval.test_case import LLMTestCaseParams
# from deepeval.test_case import LLMTestCase
from datetime import datetime
import spacy
import textstat
import spacy
from typing import Tuple, List, Set, Dict
import json


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.evaluation.benchmark_with_azure import benchmark_with_azure
from utils.evaluation.benchmark_locally import benchmark_locally
from utils.misc import save_dataset

# I need to turn off Tunnelblick to use the KCL Azure endpoint

DATASET_PATH = "data/generations/" + "5-QA-sets-2025-07-29 10:30:51.json"
MODEL_NAME = "gpt-4o-mini-chat"
LOCAL = False
CHECKPOINT = 0
CHECKPOINT_INTERVAL = 5

date = datetime.now()
date = date.strftime("%Y-%m-%d %H:%M:%S")

nlp = spacy.load("en_core_web_md")
rouge = Rouge()


def record_model_answers(dataset_path, model_name):
    print("Loading dataset")
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    with open(dataset_path, "r") as file:
        dataset = json.load(file)

    results = []

    for index, item in tqdm(
        enumerate(dataset), total=len(dataset), desc="Benchmarking model"
    ):
        question = item["question"]
        context = item["context"]
        category = item.get("category", "Uncategorized")
        expected_answer = item.get("answer", None)

        response = (benchmark_locally if LOCAL else benchmark_with_azure)(
            model_name, context, question
        )

        result_entry = {
            "Capability": "Reasoning QA",
            "Category": category,
            "Context": context,
            "Question": question,
            "Expected Answer": expected_answer,
            f"{model_name}_Response": response,
        }

        # if expected_answer is not None:
        #     result_entry["Expected Answer"] = expected_answer

        results.append(result_entry)

        if (index + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_directory = "data/model-answers/checkpoints/"
            os.makedirs(checkpoint_directory, exist_ok=True)
            checkpoint_name = (
                f"{model_name.replace('/', '_')}-{index+1}-rows-{date}.json"
            )
            with open(
                os.path.join(checkpoint_directory, checkpoint_name), "w"
            ) as json_file:
                json.dump(results, json_file, indent=4)

    output_directory = "data/model-answers/"
    os.makedirs(output_directory, exist_ok=True)
    output_path = f"{output_directory}{model_name.replace('/', '_')}-{date}.json"

    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    return results


def flatten_results_to_dataframe(results, model_name):
    flattened_data = []
    for item in results:
        if item["Capability"] == "Factual QA":
            flattened_data.append(
                {
                    "Capability": "Factual QA",
                    "Evidence": item["Evidence"],
                    "Question": item["Question"],
                    "Expected Answer": item["Expected Answer"],
                    f"{model_name}_Response": item[f"{model_name}_Response"],
                }
            )
        elif item["Capability"] == "Reasoning QA":
            for section in item["Sections"]:
                flattened_data.append(
                    {
                        "Capability": "Reasoning QA",
                        "Initial_Evidence": item["Evidence"],
                        "Evidence": section["Evidence"],
                        "Question": section["Question"],
                        "Expected Answer": section["Expected Answer"],
                        f"{model_name}_Response": section[f"{model_name}_Response"],
                    }
                )
    return pd.DataFrame(flattened_data)


def get_exact_match(expected_answer: str, model_answer: str):
    return expected_answer == model_answer


def get_f1_score(expected_answer: str, model_answer: str):
    expected_answer_tokens = expected_answer.lower().split()
    model_answer_tokens = model_answer.lower().split()

    all_tokens = list(set(expected_answer_tokens + model_answer_tokens))
    gt_vector = [1 if token in expected_answer_tokens else 0 for token in all_tokens]
    model_vector = [1 if token in model_answer_tokens else 0 for token in all_tokens]

    # Calculate precision, recall, and F1 score
    precision = np.sum(np.array(gt_vector) * np.array(model_vector)) / np.sum(
        model_vector
    )
    recall = np.sum(np.array(gt_vector) * np.array(model_vector)) / np.sum(gt_vector)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return f1


def get_sas(expected_answer: str, model_answer: str):
    doc1 = nlp(expected_answer)
    doc2 = nlp(model_answer)
    similarity = doc1.similarity(doc2)
    return similarity


def get_rouge(expected_answer: str, model_answer: str):
    scores = rouge.get_scores(expected_answer, model_answer)[0]
    return scores["rouge-1"]["f"]


def get_n_grams(expected_answer: str, model_answer: str, n: int):
    expected_n_grams = set(ngrams(expected_answer.split(), n))
    model_n_grams = set(ngrams(model_answer.split(), n))
    return len(model_n_grams.intersection(expected_n_grams))


def get_precision(expected_answer: str, model_answer: str, n: int):
    n_grams = get_n_grams(model_answer, expected_answer, n)
    total = len(list(ngrams(model_answer.split(), n)))
    if total == 0:
        return 0.0
    return n_grams / total


def get_bleu(reference: str, candidate: str):
    precisions = []
    if not isinstance(reference, list):
        reference = [reference]

    for n in range(1, 4):
        precision = sum(get_precision(candidate, ans, n) for ans in reference) / len(
            reference
        )
        precisions.append(precision)

    if precisions:
        score = np.prod(np.power(precisions, 1.0 / len(precisions)))
    else:
        score = 0
    return score


# def get_g_eval(expected_answer: str, model_answer: str):
#     # test_case = LLMTestCase(input=model_answer, actual_output=expected_answer)
#     test_case = LLMTestCase(input="Test input", actual_output="Expected output")
#     print(g_eval_metric.measure(test_case))  # Check if this returns a valid score
#     # print(f"Debug: test_case = {test_case}")
#     score = g_eval_metric.measure(test_case)
#     # print(f"Debug: score = {score}")
#     return score


def get_flesch_reading_ease(df: pd.DataFrame):
    fre_vals = []
    for _, row in df.itterows():
        question = row["Question"]
        fre_vals.append(textstat.flesch_reading_ease(question))


# We use entity recognition models (e.g., scispaCy) to extract medical terms, and calculate a relevance score: (Number of words matched concepts)/(total words).


def calculate_clinical_relevance(
    text: str, model: str = "en_core_sci_md"
) -> Tuple[float, Dict[str, Set[str]], List[str]]:
    """
    Calculate clinical term relevance score and extract medical entities from text,
    categorized by clinical domain.

    Args:
        text (str): Input text to analyze
        model (str): Name of the spaCy model to use (default: en_core_sci_md)

    Returns:
        Tuple containing:
        - float: Relevance score (matched concepts / total words)
        - Dict[str, Set[str]]: Dictionary of entity categories and their found terms
        - List[str]: List of all words in the text
    """
    try:
        nlp = spacy.load(model)
        doc = nlp(text)

        # Extract all words (excluding punctuation and whitespace)
        all_words = [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]

        # Initialize categories
        entities_by_category = {
            category: set() for category in CLINICAL_ENTITIES.keys()
        }

        # Extract and categorize entities
        for ent in doc.ents:
            for category, labels in CLINICAL_ENTITIES.items():
                if ent.label_ in labels:
                    entities_by_category[category].add(ent.text.lower())

        # Calculate total unique medical entities
        total_entities = sum(
            len(entities) for entities in entities_by_category.values()
        )

        # Calculate the relevance score
        total_words = len(all_words)
        matched_concepts = sum(
            1
            for word in all_words
            if any(word in entities for entities in entities_by_category.values())
        )

        relevance_score = matched_concepts / total_words if total_words > 0 else 0.0

        return relevance_score, entities_by_category, all_words

    except OSError as e:
        raise Exception(
            f"Failed to load spaCy model '{model}'. Please ensure it's installed: python -m spacy download {model}"
        )


def analyze_clinical_text(text: str) -> None:
    """
    Analyze clinical text and print detailed results by category.

    Args:
        text (str): Clinical text to analyze
    """
    try:
        score, entities_by_category, words = calculate_clinical_relevance(text)

        print("Clinical Text Analysis Results:")
        print(f"Overall Relevance Score: {score:.2%}")
        print(f"Total Words: {len(words)}")
        print("\nEntities by Category:")

        for category, terms in entities_by_category.items():
            if terms:  # Only show categories with found terms
                print(f"\n{category}:")
                print(f"- {', '.join(sorted(terms))}")
                print(f"- Count: {len(terms)}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")


def get_accuracy(expected_answers: list, model_answers: list) -> float:
    if len(expected_answers) != len(model_answers):
        raise ValueError("Expected and model answers must be same length")

    correct = sum(1 for exp, mod in zip(expected_answers, model_answers) if exp == mod)
    return correct / len(expected_answers)


def get_recall(expected_answer: str, model_answer: str) -> float:
    """
    Calculate recall score between expected and model answers

    Args:
        expected_answer: Expected/ground truth answer
        model_answer: Predicted/model answer

    Returns:
        float: Recall score between 0 and 1
    """
    # Convert to sets of words
    expected_words = set(expected_answer.lower().split())
    model_words = set(model_answer.lower().split())

    if len(expected_words) == 0:
        return 0.0

    # Calculate recall: correct words / total expected words
    correct_words = len(expected_words.intersection(model_words))
    return correct_words / len(expected_words)


def get_meteor(expected_answer: str, model_answer: str):
    # Tokenize the strings into words
    reference = word_tokenize(expected_answer.lower())
    hypothesis = word_tokenize(model_answer.lower())

    # Calculate the METEOR score
    score = single_meteor_score(reference, hypothesis)

    return score


# Return a dataframe containing analysis results for a given dataset


def score_model(dataset, model_name):

    if "/" in model_name:
        model_name.replace("/", "_")

    em_scores = []
    f1_scores = []
    sas_scores = []
    rouge_scores = []
    bleu_scores = []
    # clinical_concept_extraction_scores = []
    # medical_relation_extraction_scores = []
    # g_eval_scores = []

    for row in tqdm(range(0, len(dataset))):
        expected_answer = dataset["Expected Answer"][row]
        model_answer = dataset[f"{model_name} Response"][row]

        em_scores.append(get_exact_match(expected_answer, model_answer))
        f1_scores.append(get_f1_score(expected_answer, model_answer))
        sas_scores.append(get_sas(expected_answer, model_answer))
        rouge_scores.append(get_rouge(expected_answer, model_answer))
        bleu_scores.append(get_bleu(expected_answer, model_answer))

    exact_match = np.mean(em_scores)
    f1 = np.mean(f1_scores)
    semantic_answer_similarity = np.mean(sas_scores)
    rouge = np.mean(rouge_scores)
    bleu = np.mean(bleu_scores)
    # clinical_concept_extraction = np.mean(clinical_concept_extraction_scores)
    # medical_relation_extraction = np.mean(medical_relation_extraction_scores)
    # print("g_eval_scores: ", g_eval_scores)

    benchmarking_results = pd.DataFrame(
        {
            "Metric": [
                "Exact Match",
                "F1 Score",
                "Semantic Answer Similarity",
                "Rouge",
                "BLEU",
                # "Clinical Concept Extraction",
                # "Medical Relation Extraction",
                # "G-Eval",
            ],
            f"{MODEL_NAME} Score": [
                exact_match,
                f1,
                semantic_answer_similarity,
                rouge,
                bleu,
                # clinical_concept_extraction,
                # medical_relation_extraction,
                # g_eval,
            ],
        }
    )

    return benchmarking_results


def main():
    model_answers = record_model_answers(DATASET_PATH, MODEL_NAME)
    # save_dataset(model_answers, directory="model-answers")
    # model_answers = pd.read_csv("data/model-answers/Mistral-large.csv")
    # benchmarking_results = score_model(model_answers, MODEL_NAME)
    # save_dataset(benchmarking_results, directory="benchmarking-results")
    # print(benchmarking_results)


if __name__ == "__main__":
    main()

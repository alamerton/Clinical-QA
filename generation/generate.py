from datetime import datetime
import json
import sys
import os
from tqdm import tqdm
import pandas as pd
import re

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.generation.call_gpt import call_gpt  # noqa: E402
from utils.generation.call_mimic_iii import call_mimic_iii  # noqa: E402
from utils.misc import select_capability_type  # noqa: E402
from utils.generation.check_quality_with_gpt import check_quality_with_gpt  # noqa: E402
from utils.generation.reasoning_QA import (  # noqa: E402
    chunk_discharge_summary,
    create_QA_from_clinical_actions,
    create_QA_set,
    create_multistep_QA,
    segment_ds_with_llm,
    identify_clinical_actions,
)

# Dataset size
NUMBER_OF_QA_SETS: int = 5

# Control the ratio of reasoning and planning questions in the dataset
# by setting the proportion of reasoning questions. They can be any
# ratio.
FACTUAL_Q_PROPORTION: int = 0
REASONING_Q_PROPORTION: int = 1

# Variable for starting the generation from a specific row in MIMIC-III.
# Default value is 0. Set to 0 if generating new dataset.
CHECKPOINT: int = 0
CHECKPOINT_INTERVAL: int = 1

# Model for generating QA pairs
QA_GENERATION_MODEL = "gpt-4o-mini-chat"

# Model for quality-checking QA pairs
QUALITY_CHECKING_MODEL = QA_GENERATION_MODEL

# Variable for limiting the number of consecutive summaries added to the
# prompt (when multiple consecutive summaries belong to same patient).
# TODO: what is the optimal setting for this number? In the clinical setting,
# how many summaries do clinicians actually look through?
MAX_SUMMARIES: int = 1


def main():
    dataset = []

    print("Getting summaries for generation")
    discharge_summaries = call_mimic_iii(NUMBER_OF_QA_SETS, MAX_SUMMARIES)

    print("Done\n\nGenerating Q-A pairs...")

    for row in tqdm(range(CHECKPOINT, NUMBER_OF_QA_SETS)):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        discharge_summary = discharge_summaries[row]

        # Method for using mimicsid for segmentation
        chunks, full_text = chunk_discharge_summary(discharge_summary)

        # Method for using an LLM for segmentation
        # chunks = segment_ds_with_llm(
        #     capability_type, QA_GENERATION_MODEL, discharge_summary
        # )

        # Save chunks as json to view

        # with open(f"data/playground/DS_chunks-{date}.json", "w") as json_file:
        #     json.dump(chunks, json_file, indent=4)

        # Segmentation of chunks by clincal actions

        clinical_actions = identify_clinical_actions(QA_GENERATION_MODEL, chunks)

        # QA_set = create_QA_set(chunks, full_text)
        # QA_set = create_multistep_QA(chunks)

        QA_set = create_QA_from_clinical_actions(chunks, clinical_actions)

        dataset.extend(QA_set["questions"])

        print(f"{row+1}/{NUMBER_OF_QA_SETS}")

        # Save a copy of the dataset if the loop is at a checkpoint
        checkpoint_directory_path = "data/generations/checkpoints/"
        if (row + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_name = f"{row+1}-rows-{date}"
            checkpoint_path = checkpoint_directory_path + checkpoint_name
            with open(f"{checkpoint_path}.json", "w") as json_file:
                json.dump(dataset, json_file, indent=4)

    print("Complete")
    print(dataset)

    # Write dataset to output directory
    output_path = f"""data/generations/{NUMBER_OF_QA_SETS}-QA-sets-{date}"""
    with open(f"{output_path}.json", "w") as json_file:
        json.dump(dataset, json_file, indent=4)

    print("Dataset saved")


if __name__ == "__main__":
    main()

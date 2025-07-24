from zensols.nlp import FeatureToken
from zensols.mimic import Section
from zensols.mimicsid import PredictedNote, ApplicationFactory
from zensols.mimicsid.pred import SectionPredictor

from utils.generation.call_gpt import (
    call_llm_for_section_segmentation,
    call_llm_for_clinical_action_identification,
)
from utils.misc import parse_llm_segments, filter_segments


def chunk_discharge_summary(discharge_summary):
    chunks = []
    ds = discharge_summary.strip()
    section_predictor: SectionPredictor = ApplicationFactory.section_predictor()
    note: PredictedNote = section_predictor.predict([ds])[0]
    for section in note.sections.values():
        chunks.append({"id": section.id, "name": section.name, "text": section.text})
    return chunks


def segment_ds_with_llm(capability_type, model_name, discharge_summary):
    # TODO: Change output format of LLM segmentation to match that of
    # mimicsid (trello)
    chunks = []
    ds = discharge_summary.strip()
    response = call_llm_for_section_segmentation(model_name, ds, capability_type)
    batch = parse_llm_segments(response)
    for chunk in batch:
        chunks.append(chunk)
    return chunks


def identify_clinical_actions(model_name, chunks):
    """
    Passes chunks of DS data to LLM, gets clinical actions, filters
    empty and short responses, and returns list of clinical actions.
    """
    clinical_actions = []
    for chunk in chunks:
        response = call_llm_for_clinical_action_identification(model_name, chunk)
        if "name" not in response:
            continue  # skip irrelevant responses
        batch = parse_llm_segments(response)
        filtered_batch = filter_segments(batch)
        clinical_actions.extend(filtered_batch)
    return clinical_actions


def create_QA_from_clinical_actions(chunks, clinical_actions):
    full_text = "".join([chunk["text"] for chunk in chunks])

    action_spans = []
    for action in clinical_actions:
        action_text = action["text"]
        start = full_text.find(action_text)
        if start != -1:
            end = start + len(action_text)
            action_spans.append({"action": action, "start": start, "end": end})

    action_spans.sort(key=lambda x: x["start"])

    qa_pairs = []
    last_end = 0
    for span in action_spans:
        evidence = full_text[last_end : span["start"]].strip()
        if evidence:
            qa_pairs.append(
                {
                    "evidence": evidence,
                    "question": "Predict the next clinical action in the discharge summary",
                    "answer": span["action"]["text"],
                    "expected_action_name": span["action"]["name"],
                }
            )
        last_end = span["end"]

    if last_end < len(full_text):
        evidence = full_text[last_end:].strip()
        if evidence:
            qa_pairs.append(
                {
                    "evidence": evidence,
                    "question": "Predict the next clinical action in the discharge summary",
                    "answer": None,
                    "expected_action_name": None,
                }
            )

    benchmark = {
        "questions": [
            {
                "id": i,
                "context": pair["evidence"],
                "question": pair["question"],
                "answer": pair["answer"],
                "category": pair["expected_action_name"],
            }
            for i, pair in enumerate(qa_pairs)
            if pair["answer"] is not None and pair["expected_action_name"] is not None
        ]
    }
    return benchmark


def create_QA_set(chunks, full_text):
    chunk_map = {chunk["name"]: chunk for chunk in chunks}
    qa_set = []

    def add(question, context, answer, category):
        qa_set.append(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "category": category,
            }
        )

    # ------------ Knowledge and Recall ------------ #

    if "discharge-diagnosis" in chunk_map:
        add(
            "What is the discharge diagnosis?",
            full_text,
            chunk_map["discharge-diagnosis"]["text"],
            "Knowledge and Recall",
        )

    # ------------ Comprehension ------------ #

    if all(k in chunk_map for k in ["chief-complaint", "history-of-present-illness"]):
        add(
            "Why was the patient admitted?",
            f"{chunk_map['chief-complaint']['text']}\n\n"
            f"{chunk_map['history-of-present-illness']['text']}",
            chunk_map["chief-complaint"]["text"],
            "Comprehension",
        )

    if all(
        k in chunk_map
        for k in [
            "chief-complaint",
            "history-of-present-illness",
            "assessment-and-plan",
        ]
    ):
        add(
            "What is the most effective way to get an accurate diagnosis?",
            f"{chunk_map['chief-complaint']['text']}\n\n"
            f"{chunk_map['history-of-present-illness']['text']}",
            chunk_map["assessment-and-plan"]["text"],
            "Comprehension",
        )

    # Test evidence and reasoning
    if any(
        k in chunk_map
        for k in [
            "patient-test-information",
            "labs",
            "labs-imaging",
            "imaging",
            "findings",
            "wet-read",
        ]
    ):
        test_keys = [
            k
            for k in [
                "patient-test-information",
                "labs",
                "labs-imaging",
                "imaging",
                "findings",
                "wet-read",
            ]
            if k in chunk_map
        ]
        combined_text = "\n\n".join(chunk_map[k]["text"] for k in test_keys)

        add(
            "Were the tests important? If so, provide evidence and reasoning why.",
            full_text,
            combined_text,
            "Comprehension",
        )

    # ------------ Application and Analysis ------------ #

    if all(k in chunk_map for k in ["discharge-diagnosis", "followup-instructions"]):
        add(
            "What follow-up is appropriate for this diagnosis?",
            f"{chunk_map['discharge-diagnosis']['text']}",
            chunk_map["followup-instructions"]["text"],
            "Application and Analysis",
        )

    # ------------ Synthesis and Evaluation ------------ #

    if all(k in chunk_map for k in ["hospital-course", "discharge-diagnosis"]):
        add(
            "Evaluate the effectiveness of the treatment given.",
            f"{chunk_map['hospital-course']['text']}\n\n"
            f"{chunk_map['discharge-diagnosis']['text']}",
            "N/A",
            "Synthesis and Evaluation",
        )

    return qa_set

    # ------------ Next Clinical Steps Multi-Step ------------ #

    # initial chunks until hospital course

    # what is the expected hospital course?

    # hospital course

    # what are the expected discharge medications?

    # discharge medications

    # next parts

    # what is the expected discharge diagnosis?

    # discharge diagnosis

    # next parts

    # what are the expected discharge instructions?


def create_multistep_QA(chunks):
    chunk_map = {chunk["name"]: chunk for chunk in chunks}
    qa_steps = []

    def step(question, prompt_chunks, answer_chunk, category):
        context = "\n\n".join(
            chunk_map[k]["text"] for k in prompt_chunks if k in chunk_map
        )
        if answer_chunk in chunk_map:
            answer = chunk_map[answer_chunk]["text"]

            qa_steps.append(
                {
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "category": category,
                }
            )

    # Step 1: Predict assessment and plan from earlier info
    step(
        "What is the expected assessment and plan?",
        ["chief-complaint", "history-of-present-illness"],
        "assessment-and-plan",
        "Next Clinical Steps",
    )

    # Step 2: Predict hospital course
    step(
        "What is the expected hospital course?",
        ["chief-complaint", "history-of-present-illness", "assessment-and-plan"],
        "hospital-course",
        "Next Clinical Steps",
    )

    # Step 2: Predict discharge medications
    step(
        "What are the expected discharge medications?",
        [
            "chief-complaint",
            "history-of-present-illness",
            "assessment-and-plan",
            "hospital-course",
        ],
        "discharge-medications",
        "Next Clinical Steps",
    )

    # Step 3: Predict discharge diagnosis
    step(
        "What is the expected discharge diagnosis?",
        [
            "chief-complaint",
            "history-of-present-illness",
            "assessment-and-plan",
            "hospital-course",
        ],
        "discharge-diagnosis",
        "Next Clinical Steps",
    )

    # Step 4: Predict follow-up instructions
    step(
        "What are the expected discharge instructions?",
        [
            "chief-complaint",
            "history-of-present-illness",
            "assessment-and-plan",
            "hospital-course",
            "discharge-diagnosis",
        ],
        "discharge-instructions",
        "Next Clinical Steps",
    )

    return qa_steps


# def create_multistep_QA_leave_missing(chunks):
#     chunk_map = {chunk["name"]: chunk for chunk in chunks}
#     qa_steps = []
#     step_id = 1

#     sequence = [
#         {
#             "question": "What is the expected hospital course?",
#             "prompt_keys": [
#                 "chief-complaint",
#                 "history-of-present-illness",
#                 "assessment-and-plan",
#             ],
#             "answer_key": "hospital-course",
#         },
#         {
#             "question": "What are the expected discharge medications?",
#             "prompt_keys": [
#                 "chief-complaint",
#                 "history-of-present-illness",
#                 "assessment-and-plan",
#                 "hospital-course",
#             ],
#             "answer_key": "discharge-medications",
#         },
#         {
#             "question": "What is the expected discharge diagnosis?",
#             "prompt_keys": [
#                 "chief-complaint",
#                 "history-of-present-illness",
#                 "assessment-and-plan",
#                 "hospital-course",
#             ],
#             "answer_key": "discharge-diagnosis",
#         },
#         {
#             "question": "What are the expected discharge instructions?",
#             "prompt_keys": [
#                 "chief-complaint",
#                 "history-of-present-illness",
#                 "assessment-and-plan",
#                 "hospital-course",
#                 "discharge-diagnosis",
#             ],
#             "answer_key": "followup-instructions",
#         },
#     ]

#     for step in sequence:
#         required = step["prompt_keys"] + [step["answer_key"]]
#         if not all(k in chunk_map for k in required):
#             break

#         context = "\n\n".join(chunk_map[k]["text"] for k in step["prompt_keys"])
#         answer = chunk_map[step["answer_key"]]["text"]

#         qa_steps.append(
#             {
#                 "step_id": step_id,
#                 "question": step["question"],
#                 "context": context,
#                 "expected_answer": answer,
#                 "category": "Next Clinical Steps",
#             }
#         )
#         step_id += 1

#     return qa_steps

from zensols.nlp import FeatureToken
from zensols.mimic import Section
from zensols.mimicsid import PredictedNote, ApplicationFactory
from zensols.mimicsid.pred import SectionPredictor


def chunk_discharge_summary(discharge_summary):
    chunks = []
    ds = discharge_summary.strip()
    section_predictor: SectionPredictor = ApplicationFactory.section_predictor()
    note: PredictedNote = section_predictor.predict([ds])[0]
    for section in note.sections.values():
        chunks.append({"id": section.id, "name": section.name, "text": section.text})
    return chunks, ds


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
        answer = chunk_map[answer_chunk]["text"] if answer_chunk in chunk_map else "N/A"
        qa_steps.append(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "category": category,
            }
        )

    # Step 1: Predict hospital course from earlier info
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
            "discharge-medications",
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
            "discharge-medications",
            "discharge-diagnosis",
        ],
        "followup-instructions",
        "Next Clinical Steps",
    )

    return qa_steps

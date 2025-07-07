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
    return chunks


def create_QA_set(chunks):
    chunk_map = {chunk["name"]: chunk for chunk in chunks}
    qa_set = []

    # Rule 1: HPI + Hospital Course -> Discharge Diagnosis
    if all(
        k in chunk_map
        for k in [
            "history-of-present-illness",
            "hospital-course",
            "discharge-diagnosis",
        ]
    ):
        qa_set.append(
            {
                "question": "Given the history of present illness and hospital course, what is the discharge diagnosis?",
                "context": (
                    f"History of Present Illness: {chunk_map['history-of-present-illness']['text']}\n\n"
                    f"Hospital Course: {chunk_map['hospital-course']['text']}"
                ),
                "answer": chunk_map["discharge-diagnosis"]["text"],
            }
        )

    # Rule 2: Discharge Diagnosis -> Discharge Medications
    if all(k in chunk_map for k in ["discharge-diagnosis", "discharge-medications"]):
        qa_set.append(
            {
                "question": "Given the discharge diagnosis, what medications were prescribed at discharge?",
                "context": f"Discharge Diagnosis: {chunk_map['discharge-diagnosis']['text']}",
                "answer": chunk_map["discharge-medications"]["text"],
            }
        )

    # Rule 3: Chief Complaint + HPI -> Hospital Course
    if all(
        k in chunk_map
        for k in ["chief-complaint", "history-of-present-illness", "hospital-course"]
    ):
        qa_set.append(
            {
                "question": "Based on the chief complaint and history of present illness, what was the hospital course?",
                "context": (
                    f"Chief Complaint: {chunk_map['chief-complaint']['text']}\n\n"
                    f"History of Present Illness: {chunk_map['history-of-present-illness']['text']}"
                ),
                "answer": chunk_map["hospital-course"]["text"],
            }
        )

    # Rule 4: Discharge Diagnosis -> Followup Instructions
    if all(k in chunk_map for k in ["discharge-diagnosis", "followup-instructions"]):
        qa_set.append(
            {
                "question": "Based on the discharge diagnosis, what follow-up instructions were given?",
                "context": f"Discharge Diagnosis: {chunk_map['discharge-diagnosis']['text']}",
                "answer": chunk_map["followup-instructions"]["text"],
            }
        )

    # Rule 5: Hospital Course -> Procedures
    if all(k in chunk_map for k in ["hospital-course", "procedures"]):
        qa_set.append(
            {
                "question": "What procedures were performed during the hospital course?",
                "context": f"Hospital Course: {chunk_map['hospital-course']['text']}",
                "answer": chunk_map["procedures"]["text"],
            }
        )

    return qa_set

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

    # Knowledge and Recall
    if "discharge-diagnosis" in chunk_map:
        add(
            "What is the discharge diagnosis?",
            full_text,
            chunk_map["discharge-diagnosis"]["text"],
            "Knowledge and Recall",
        )

    # Comprehension
    if all(k in chunk_map for k in ["chief-complaint", "history-of-present-illness"]):
        add(
            "Why was the patient admitted?",
            f"Chief Complaint: {chunk_map['chief-complaint']['text']}\n\n"
            f"History of Present Illness: {chunk_map['history-of-present-illness']['text']}",
            chunk_map["history-of-present-illness"]["text"],
            "Comprehension",
        )

    # Application and Analysis
    if all(k in chunk_map for k in ["discharge-diagnosis", "followup-instructions"]):
        add(
            "What follow-up is appropriate for this diagnosis?",
            f"Discharge Diagnosis: {chunk_map['discharge-diagnosis']['text']}",
            chunk_map["followup-instructions"]["text"],
            "Application and Analysis",
        )

    # Synthesis and Evaluation
    if all(k in chunk_map for k in ["hospital-course", "discharge-diagnosis"]):
        add(
            "Evaluate the effectiveness of the treatment given.",
            f"Hospital Course: {chunk_map['hospital-course']['text']}\n\n"
            f"Discharge Diagnosis: {chunk_map['discharge-diagnosis']['text']}",
            chunk_map["discharge-diagnosis"]["text"],
            "Synthesis and Evaluation",
        )

    return qa_set

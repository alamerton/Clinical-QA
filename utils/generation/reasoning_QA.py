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
    # if 'allergies' in chunk:
    #   if 'chief-complaint' in chunk:
    #       do logic...

    return

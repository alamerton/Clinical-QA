from zensols.nlp import FeatureToken
from zensols.mimic import Section
from zensols.mimicsid import PredictedNote, ApplicationFactory
from zensols.mimicsid.pred import SectionPredictor


def chunk_discharge_summary(discharge_summary):
    section_predictor: SectionPredictor = ApplicationFactory.section_predictor()
    ds = discharge_summary.strip()
    note: PredictedNote = section_predictor.predict(ds)
    # sec: Section
    # chunks = []
    # for sec in note.sections.values():
    #     chunks.append(sec.id, se)
    print(note)
    return 0


def create_QA_set(chunks):
    return

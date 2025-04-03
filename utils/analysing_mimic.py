import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any


def load_discharge_summaries(filepath: str) -> pd.DataFrame:
    """
    Load discharge summaries from MIMIC III noteevents, filtering only for discharge summary types.

    Args:
        filepath (str): Path to the MIMIC III noteevents CSV file

    Returns:
        pandas.DataFrame: Filtered DataFrame of discharge summaries
    """
    df = pd.read_csv(filepath)
    discharge_summaries = df[df["CATEGORY"] == "Discharge summary"]
    return discharge_summaries


def analyze_document_structure(summaries: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze structural patterns in discharge summaries.

    Args:
        summaries (pd.DataFrame): DataFrame of discharge summaries

    Returns:
        Dict containing various structural insights
    """
    structural_analysis = {
        "total_summaries": len(summaries),
        "unique_patients": summaries["SUBJECT_ID"].nunique(),
        "avg_summary_length": summaries["TEXT"].str.len().mean(),
        "section_patterns": identify_common_sections(summaries),
        "temporal_sections": identify_temporal_sections(summaries),
    }

    return structural_analysis


def identify_common_sections(summaries: pd.DataFrame) -> List[Dict[str, float]]:
    """
    Identify common section headers and their frequencies.

    Args:
        summaries (pd.DataFrame): DataFrame of discharge summaries

    Returns:
        List of dictionaries with section headers and their frequencies
    """
    # Common medical discharge summary section patterns
    section_patterns = [
        r"ADMISSION DATE:",
        r"DISCHARGE DATE:",
        r"CHIEF COMPLAINT:",
        r"HISTORY OF PRESENT ILLNESS:",
        r"PAST MEDICAL HISTORY:",
        r"SOCIAL HISTORY:",
        r"FAMILY HISTORY:",
        r"MEDICATIONS:",
        r"ALLERGIES:",
        r"PHYSICAL EXAMINATION:",
        r"LABORATORY DATA:",
        r"DIAGNOSTIC STUDIES:",
        r"HOSPITAL COURSE:",
        r"PROCEDURES:",
        r"CONSULTATIONS:",
        r"DISCHARGE DIAGNOSES:",
        r"DISCHARGE CONDITION:",
        r"DISCHARGE MEDICATIONS:",
        r"DISCHARGE INSTRUCTIONS:",
    ]

    section_frequencies = []
    for pattern in section_patterns:
        frequency = summaries["TEXT"].str.contains(pattern, case=False).mean()
        section_frequencies.append(
            {"section": pattern.replace(":", ""), "frequency": frequency}
        )

    return sorted(section_frequencies, key=lambda x: x["frequency"], reverse=True)


def identify_temporal_sections(summaries: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify sections with strong temporal/chronological indicators.

    Args:
        summaries (pd.DataFrame): DataFrame of discharge summaries

    Returns:
        Dictionary of temporal sections
    """
    temporal_indicators = {
        "admission_timeline": ["on admission", "initial presentation", "when admitted"],
        "hospital_course": [
            "during hospitalization",
            "hospital course",
            "while in hospital",
        ],
        "progression": ["subsequently", "thereafter", "following", "then", "next"],
        "daily_progress": ["hospital day 1", "hospital day 2", "post-operative day"],
    }

    return temporal_indicators


def visualize_section_frequencies(section_patterns):
    """
    Create a visualization of section frequencies.

    Args:
        section_patterns (List[Dict]): List of section frequency dictionaries
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    sections = [s["section"] for s in section_patterns]
    frequencies = [s["frequency"] for s in section_patterns]

    plt.bar(sections, frequencies)
    plt.title("Discharge Summary Section Frequencies")
    plt.xlabel("Sections")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main(filepath: str):
    """
    Main analysis function.

    Args:
        filepath (str): Path to MIMIC III noteevents CSV
    """
    # Load discharge summaries
    summaries = load_discharge_summaries(filepath)

    # Perform structural analysis
    structural_analysis = analyze_document_structure(summaries)

    # Print results
    print("Structural Analysis of Discharge Summaries:")
    for key, value in structural_analysis.items():
        print(f"{key}: {value}")

    # Visualize section frequencies
    visualize_section_frequencies(structural_analysis["section_patterns"])


# Example usage
main("/Users/alfielamerton/Documents/Code/C-QuAL/mimic_discharge_summaries.csv")

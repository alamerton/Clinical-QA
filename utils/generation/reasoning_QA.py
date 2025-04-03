import re


def extract_discharge_summary_sections(discharge_summary):
    """
    Extract sections from a MIMIC-III discharge summary using regex pattern matching.
    Returns a list of dictionaries with section names and content.
    """
    # Common section headers in MIMIC-III discharge summaries
    section_patterns = [
        r"CHIEF COMPLAINT:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"HISTORY OF PRESENT ILLNESS:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"PAST MEDICAL HISTORY:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"ADMISSION MEDICATIONS:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"ALLERGIES:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"PHYSICAL EXAMINATION:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"LABORATORY DATA:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"HOSPITAL COURSE:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"MEDICATIONS ON DISCHARGE:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"DISCHARGE DISPOSITION:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"DISCHARGE DIAGNOSIS:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"DISCHARGE CONDITION:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"PLAN:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"ASSESSMENT:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"OPERATIONS AND PROCEDURES:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"BRIEF HOSPITAL COURSE:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
        r"DISCHARGE INSTRUCTIONS:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)",
    ]

    # Alternative approach: find all section headers first, then extract sections
    section_header_pattern = r"\n([A-Z][A-Z ]*):"
    section_headers = re.findall(section_header_pattern, discharge_summary)

    sections = []

    # Use the common patterns first
    for pattern in section_patterns:
        section_name = re.search(r"([A-Z]+ ?[A-Z]+):", pattern).group(1)
        matches = re.search(pattern, discharge_summary, re.DOTALL)
        if matches:
            content = matches.group(1).strip()
            sections.append({"name": section_name, "content": content})

    # For any sections not captured by the patterns above, try to extract them
    for header in section_headers:
        if not any(section["name"] == header for section in sections):
            pattern = rf"\n{header}:(.*?)(?=\n[A-Z]+ ?[A-Z]+:|\Z)"
            matches = re.search(pattern, discharge_summary, re.DOTALL)
            if matches:
                content = matches.group(1).strip()
                sections.append({"name": header, "content": content})

    return sections


def get_initial_presentation(sections):
    """
    Extract the initial presentation information from the discharge summary sections.
    Prioritizes sections like Chief Complaint and History of Present Illness.
    """
    initial_sections = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS"]
    presentation_text = ""

    for section_name in initial_sections:
        for section in sections:
            if section["name"] == section_name:
                presentation_text += f"{section_name}:\n{section['content']}\n\n"

    # If we couldn't find specific initial sections, use any admission-related section
    if not presentation_text:
        for section in sections:
            if "ADMISSION" in section["name"] or "PRESENTING" in section["name"]:
                presentation_text += f"{section['name']}:\n{section['content']}\n\n"

    # As a fallback, just use the first available section
    if not presentation_text and sections:
        presentation_text = f"{sections[0]['name']}:\n{sections[0]['content']}"

    return presentation_text.strip()


def get_subsequent_steps(sections):
    """
    Extract the subsequent clinical steps from discharge summary sections.
    Orders sections to represent the clinical course chronologically.
    """
    # Define priority and chronological order of sections
    clinical_progression = [
        "PHYSICAL EXAMINATION",
        "LABORATORY DATA",
        "ASSESSMENT",
        "PLAN",
        "HOSPITAL COURSE",
        "BRIEF HOSPITAL COURSE",
        "OPERATIONS AND PROCEDURES",
        "MEDICATIONS ON DISCHARGE",
        "DISCHARGE DIAGNOSIS",
        "DISCHARGE DISPOSITION",
        "DISCHARGE CONDITION",
        "DISCHARGE INSTRUCTIONS",
    ]

    subsequent_steps = []

    # First add sections in the expected clinical order
    for section_name in clinical_progression:
        for section in sections:
            if section["name"] == section_name:
                # For longer sections like HOSPITAL COURSE, we may want to split further
                if section_name in ["HOSPITAL COURSE", "BRIEF HOSPITAL COURSE"]:
                    # Try to break hospital course into subsections or paragraphs
                    sub_sections = break_into_subsections(section["content"])
                    for sub_section in sub_sections:
                        subsequent_steps.append(
                            {
                                "name": f"{section_name} - {sub_section.get('name', 'Part')}",
                                "content": sub_section["content"],
                            }
                        )
                else:
                    subsequent_steps.append(section)

    # Add any remaining sections not explicitly ordered
    section_names_added = [step["name"] for step in subsequent_steps]
    for section in sections:
        if section["name"] not in section_names_added and section["name"] not in [
            "CHIEF COMPLAINT",
            "HISTORY OF PRESENT ILLNESS",
        ]:
            subsequent_steps.append(section)

    return subsequent_steps


def break_into_subsections(text):
    """
    Attempts to break a longer section like Hospital Course into logical subsections
    based on paragraph breaks, numbered lists, or specific keywords.
    """
    subsections = []

    # Try to find numbered items (1. 2. etc.)
    numbered_items = re.findall(r"(\d+\.\s*[^\d].*?)(?=\d+\.\s*|\Z)", text, re.DOTALL)

    if numbered_items:
        for i, item in enumerate(numbered_items):
            subsections.append({"name": f"Step {i+1}", "content": item.strip()})
        return subsections

    # Try to find subsection headings (often capitalized phrases followed by colon)
    subsection_matches = re.findall(
        r"([A-Z][A-Za-z\s]+:)(.*?)(?=[A-Z][A-Za-z\s]+:|\Z)", text, re.DOTALL
    )

    if subsection_matches:
        for heading, content in subsection_matches:
            subsections.append({"name": heading.strip(":"), "content": content.strip()})
        return subsections

    # Fall back to paragraph breaks
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) > 1:
        for i, para in enumerate(paragraphs):
            subsections.append({"name": f"Paragraph {i+1}", "content": para.strip()})
        return subsections

    # If we can't break it down, return the whole section
    subsections.append({"name": "Complete", "content": text.strip()})

    return subsections

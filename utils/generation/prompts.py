def get_reasoning_generation_prompt(discharge_summary_string):
    return (
        """You are an expert medical annotator tasked with creating a clinical planning evaluation using a discharge summary from the MIMIC-III database. """,
        f"""Your goal is to extract, structure, and generate questions that test an LLM's ability to simulate clinical reasoning and planning.

        Part 1:
        - Identify and extract the initial clinical presentation section that includes:
            - Primary reason for admission
            - Key presenting symptoms
            - Critical background information
            - Initial vital signs and relevant lab values
        - This section should represent the initial decision point where clinical reasoning begins
        - Extract only information available at admission, before any interventions

        Part 2:
        - Extract ALL subsequent clinical information in chronological order, including:
            - Diagnostic procedures performed
            - Treatment decisions and modifications
            - Clinical findings and results
            - Patient response to interventions
            - Final outcomes
        - Organize this information to show the progression of clinical decision-making

        Part 3:
        Generate sequential reasoning questions that:
        - Start with ONLY the information from Part 1
        - For each major clinical decision or finding in Part 2:
            Q: What would be the next appropriate clinical step?
            A: [Actual step taken from Part 2]
            Reasoning: [Clinical logic connecting Part 1 to this decision]

        Format your response EXACTLY as:
        'Initial_Presentation: $part1
        Subsequent_Course: $part2
        Clinical_Reasoning_Questions: $part3'
        (without quotes) where $part1 is just the information requested in Part 1, $part2 is just the information requested in Part 2, and $part3 is just the information requested in Part 3. DO NOT ADD ANY OTHER TEXT OR PREAMBLE.

        Discharge Summary: {discharge_summary_string}
        """,
    )


def get_factual_generation_prompt(question_type, discharge_summary_string):
    return (
        """You are a medical expert tasked with creating a sophisticated clinical reasoning benchmark using a discharge summary from the MIMIC-III database. Your objective is to design an assessment that captures the nuanced clinical decision-making process.""",
        f"""Your task is to generate two critical components:

            Part 1:

            - Construct a question that:
                - Directly reflects the key diagnostic or treatment reasoning in the discharge summary
                - Requires multi-step clinical inference
                - Cannot be answered by simple fact retrieval
                - Challenges the deep understanding of medical context
                - Uses language that mimics authentic clinical reasoning
                - Is of the following question type: {question_type}

            Part 2:

            - Provide a concise, precise answer that:
                - Demonstrates the specific clinical reasoning pathway
                - Reflects the exact decision-making process used by the original clinician
                - Is evidence-based and directly traceable to the discharge summary

            Guidance:

            - The question should be sufficiently complex to differentiate between surface-level information processing and genuine clinical reasoning
            - Avoid questions that can be answered through simple pattern matching
            - Prioritise questions that require hypothesis generation, risk assessment, or complex diagnostic inference

            
            Please follow this format EXACTLY:
            
            Part 1: $part1\n
            Part 2: $part2\n

            Where $part1 is just the question requested in part 1, and $part2 is just the answer in part 2. DO NOT ADD ANY OTHER TEXT. DO NOT INCLUDE ANY PREAMBLE.

            Discharge Summary: {discharge_summary_string}
        """,
    )


def get_reasoning_qual_check_prompt(qa_string):
    return (
        """You are a senior medical expert responsible for critically evaluating a clinical reasoning benchmark question-answer pair generated from a discharge summary.""",
        f""" Evaluation Criteria:
            1. Clinical Reasoning Depth (40%)
            - Does the question require sophisticated clinical inference?
            - Does it test genuine medical decision-making beyond surface-level information?
            - Can the question not be answered through simple fact retrieval?

            2. Evidence Alignment (30%)
            - Is the provided evidence directly relevant to answering the question?
            - Does the evidence support a meaningful clinical reasoning process?
            - Are the evidence chunks appropriately selected and crucial to the reasoning?

            3. Question Quality (20%)
            - Is the question formulated in a clinically authentic manner?
            - Does it avoid directly revealing the answer?
            - Is the question sufficiently challenging and nuanced?

            4. Answer Accuracy (10%)
            - Does the answer precisely reflect the clinical reasoning in the discharge summary?
            - Is the answer concise and evidence-based?

            Scoring Guidance:
            - Score 1: Meets all criteria exceptionally well
            - Score 0: Fails to meet one or more critical criteria

            Please only respond with either the number 0 or 1, representing the score.

            Output Format:
            $score 

            Where $score is 0 or 1. DO NOT ADD ANY OTHER TEXT. DO NOT INCLUDE ANY PREAMBLE.

            Here is the question-answer pair for you to work on:
            
            {qa_string}
        """,
    )


def get_factual_qual_check_prompt(qa_string):
    return (
        """You are a senior medical expert responsible for critically evaluating a clinical reasoning benchmark question-answer pair generated from a discharge summary.""",
        f"""Evaluation Criteria:
            1. Initial Scenario Comprehensiveness (35%)
            - Does the initial section provide sufficient context for clinical reasoning?
            - Are all critical patient details included?
            - Can a skilled clinician develop meaningful hypotheses from this information?

            2. Subsequent Course Revelation (35%)
            - Does the subsequent information demonstrate the actual clinical decision-making process?
            - Are the diagnostic and treatment steps clearly and logically presented?
            - Does the scenario reveal the complexity of medical problem-solving?

            3. Reasoning Trajectory (20%)
            - Is there a clear evolution of clinical thinking?
            - Do the initial and subsequent sections create a meaningful narrative of medical decision-making?

            4. Educational Value (10%)
            - Would this scenario be useful for testing an LLM's clinical reasoning abilities?
            - Does it capture nuanced medical decision-making?

            Scoring Guidance:
            - Score 1: Exceptionally strong scenario that comprehensively demonstrates clinical reasoning
            - Score 0: Fails to meet one or more critical criteria

            Please only respond with either the number 0 or 1, representing the score.

            Output Format:
            $score 

            Where $score is 0 or 1. DO NOT ADD ANY OTHER TEXT. DO NOT INCLUDE ANY PREAMBLE.
        
            Here is the question-answer pair for you to work on:
            
            {qa_string}
        """,
    )


def get_segmentation_prompt(discharge_summary_string):
    return (
        """You are an expert medical annotator tasked with segmenting clinical discharge summary from the MIMIC-III database. """,
        f"""
            You are a clinical language model trained for precise document structuring. Segment the following MIMIC-III discharge summary into its standard sections. Identify and extract each section using canonical headers (e.g., "Chief Complaint", "History of Present Illness", "Past Medical History", "Medications on Admission", "Hospital Course", "Discharge Medications", "Discharge Diagnosis", "Follow-up Instructions", "Allergies", etc.). Return the result as a dictionary with section headers as keys and corresponding section text as values. If a section is missing, omit the key entirely. Do not hallucinate content. Use only the text provided.

            Input Discharge Summary:
            \"\"\"{discharge_summary_string}\"\"\"

            Output Format:
            {{
            "Chief Complaint": "...",
            "History of Present Illness": "...",
            ...
            }}
            """,
    )

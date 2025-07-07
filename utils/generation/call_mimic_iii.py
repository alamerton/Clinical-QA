import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

database_host = os.environ.get("DATABASE_HOST")
database_username = os.environ.get("DATABASE_USERNAME")
database_password = os.environ.get("DATABASE_PASSWORD")
database_name = os.environ.get("DATABASE_NAME")
database_port = os.environ.get("DATABASE_PORT")

# Variable to flag where to save collected discharge summaries, default
# is "function".
SUMMARIES_DESTINATION = "function"


# Load query results into a dataframe containing subject_ids and notes
def save_data(results):
    results_df = pd.DataFrame(results, columns=["subject_id", "note"])
    results_df.to_csv("C-QuAL/data/mimic-iii-subset.csv")


# Add a start and end tag to each discharge summary for the LLM to
# identify and concatenate them.
def prepare_discharge_summaries(discharge_summaries):
    multiple_summaries = ""
    for i in range(0, len(discharge_summaries)):
        start_string = f"[Discharge summary {i + 1} start]\n"
        end_string = f"\n[Discharge summary {i + 1} end]\n"
        discharge_summary = discharge_summaries[i]
        multiple_summaries += start_string + discharge_summary + end_string
    return multiple_summaries


# Return a discharge summary from the MIMIC-III database
def call_mimic_iii(num_rows, max_summaries):
    # Get date and time data for outputs
    current_date = datetime.now().date()

    # Set up DB connection
    connection = psycopg2.connect(
        host=database_host,
        user=database_username,
        password=database_password,
        database=database_name,
        port=database_port,
    )
    cursor = connection.cursor()

    discharge_summaries = []

    query = """
        SELECT subject_id, text 
        FROM public.noteevents
        ORDER BY row_id ASC 
        LIMIT %s;
    """

    cursor.execute(query, (num_rows,))
    rows = cursor.fetchall()

    current_subject_id = None
    current_summaries = []

    # For each row in array of discharge summaries
    for i, row in enumerate(rows):
        subject_id, discharge_summary = row

        if subject_id != current_subject_id:
            if current_summaries:
                if len(current_summaries) > 1:
                    combined_summaries = prepare_discharge_summaries(current_summaries)
                    discharge_summaries.extend(
                        [combined_summaries] * len(current_summaries)
                    )
                else:
                    discharge_summaries.append(current_summaries[0])
            current_subject_id = subject_id
            current_summaries = [discharge_summary]
        else:
            if len(current_summaries) < max_summaries:
                current_summaries.append(discharge_summary)
            else:
                if len(current_summaries) > 1:
                    combined_summaries = prepare_discharge_summaries(current_summaries)
                    discharge_summaries.extend(
                        [combined_summaries] * len(current_summaries)
                    )
                else:
                    discharge_summaries.append(current_summaries[0])
                current_summaries = [discharge_summary]

    # Handle final batch after loop
    if current_summaries:
        if len(current_summaries) > 1:
            combined_summaries = prepare_discharge_summaries(current_summaries)
            discharge_summaries.append(combined_summaries)

        else:
            discharge_summaries.append(current_summaries[0])

    # Close the database connection
    cursor.close()
    connection.close()

    if current_summaries:
        if len(current_summaries) > 1:
            combined_summaries = prepare_discharge_summaries(current_summaries)
            discharge_summaries.append(combined_summaries)
        else:
            single_summary = current_summaries[0]
            discharge_summaries.append(single_summary)

    if SUMMARIES_DESTINATION == "file":  # Save discharge summaries to file
        with open(f"data/{num_rows}-discharge-summaries-{current_date}.json", "w") as f:
            json.dump(discharge_summaries, f)

    elif (
        SUMMARIES_DESTINATION == "function"
    ):  # Send discharge summaries to calling function
        return discharge_summaries

    else:
        raise ValueError("Destination value must be either 'file' or 'function'")

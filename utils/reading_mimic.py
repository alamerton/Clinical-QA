# from generation.call_mimic_iii import call_mimic_iii

# import pandas as pd
# import json
# import re
# from typing import List, Optional


# def process_mimic_noteevents(
#     discharge_summaries: List[str],
#     output_format: str = "csv",
#     output_path: Optional[str] = None,
# ) -> pd.DataFrame:
#     """
#     Process MIMIC-III discharge summaries into a structured DataFrame.

#     Parameters:
#     -----------
#     discharge_summaries : List[str]
#         List of discharge summaries from the MIMIC-III database
#     output_format : str, optional (default='csv')
#         Output format. Options: 'csv', 'json', 'excel'
#     output_path : str, optional
#         Path to save the output file. If None, returns DataFrame

#     Returns:
#     --------
#     pd.DataFrame
#         Processed DataFrame with structured discharge summary data
#     """
#     # Create a list to store processed summaries
#     processed_summaries = []

#     # Process each summary
#     for summary in discharge_summaries:
#         # Check if summary contains multiple discharge summary markers
#         if "[Discharge summary" in summary:
#             # Use regex to split and extract summaries
#             summary_matches = re.findall(
#                 r"\[Discharge summary \d+ start\]\n(.*?)\n\[Discharge summary \d+ end\]",
#                 summary,
#                 re.DOTALL,
#             )

#             for content in summary_matches:
#                 processed_summaries.append(
#                     {
#                         "text": content.strip(),
#                     }
#                 )
#         else:
#             # Single summary processing
#             processed_summaries.append(
#                 {
#                     "text": summary,
#                 }
#             )

#     # Convert to DataFrame
#     df = pd.DataFrame(processed_summaries)


#     # Output handling
#     if output_path:
#         if output_format == "csv":
#             df.to_csv(output_path, index=False)
#         elif output_format == "json":
#             df.to_json(output_path, orient="records")
#         elif output_format == "excel":
#             df.to_excel(output_path, index=False)
#         print(f"Data saved to {output_path}")

#     return df


# def analyze_mimic_summaries(df: pd.DataFrame) -> dict:
#     """
#     Perform basic statistical analysis on the discharge summaries.

#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame of processed discharge summaries

#     Returns:
#     --------
#     dict
#         Dictionary of summary statistics
#     """
#     analysis = {
#         "total_summaries": len(df),
#         "total_characters": df["length"].sum(),
#         "average_summary_length": df["length"].mean(),
#         "average_word_count": df["word_count"].mean(),
#         "longest_summary": (
#             df.loc[df["length"].idxmax(), "text"][:200] + "..."
#             if len(df) > 0
#             else "No summaries"
#         ),
#         "shortest_summary": (
#             df.loc[df["length"].idxmin(), "text"][:200] + "..."
#             if len(df) > 0
#             else "No summaries"
#         ),
#     }

#     return analysis


# # Example usage
# def main():
#     # Assuming you've already imported or defined call_mimic_iii function
#     # from your original script

#     # Process summaries
#     discharge_summaries = call_mimic_iii(num_rows=1000, max_summaries=2)

#     # Process summaries
#     df = process_mimic_noteevents(
#         discharge_summaries,
#         output_format="csv",
#         output_path="mimic_discharge_summaries.csv",
#     )

#     # Analyze summaries
#     summary_stats = analyze_mimic_summaries(df)

#     # Print analysis
#     print("Summary Statistics:")
#     for key, value in summary_stats.items():
#         print(f"{key}: {value}")


# if __name__ == "__main__":
#     main()

import pandas as pd
import re
from typing import List, Optional


def process_mimic_noteevents(
    discharge_summaries: List[str],
    output_format: str = "csv",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract only discharge summary data from MIMIC-III noteevents.

    Parameters:
    -----------
    discharge_summaries : List[str]
        List of discharge summaries from the MIMIC-III database
    output_format : str, optional (default='csv')
        Output format. Options: 'csv', 'json', 'excel'
    output_path : str, optional
        Path to save the output file. If None, returns DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame containing only discharge summary data
    """
    # Create a list to store processed summaries
    processed_summaries = []

    # Process each summary
    for summary in discharge_summaries:
        # Check if summary contains multiple discharge summary markers
        if "[Discharge summary" in summary:
            # Use regex to split and extract summaries
            summary_matches = re.findall(
                r"\[Discharge summary \d+ start\]\n(.*?)\n\[Discharge summary \d+ end\]",
                summary,
                re.DOTALL,
            )

            for content in summary_matches:
                processed_summaries.append(content.strip())
        else:
            # Single summary processing
            processed_summaries.append(summary.strip())

    # Convert to DataFrame with single column
    df = pd.DataFrame(processed_summaries, columns=["discharge_summary"])

    # Output handling
    if output_path:
        if output_format == "csv":
            df.to_csv(output_path, index=False)
        elif output_format == "json":
            df.to_json(output_path, orient="records")
        elif output_format == "excel":
            df.to_excel(output_path, index=False)
        print(f"Data saved to {output_path}")

    return df


# Example usage
def main():
    # Assuming you've already imported or defined call_mimic_iii function
    # from your original script

    from generation.call_mimic_iii import call_mimic_iii

    # Process summaries
    discharge_summaries = call_mimic_iii(num_rows=1000, max_summaries=2)

    # Extract and save discharge summaries
    df = process_mimic_noteevents(
        discharge_summaries,
        output_format="csv",
        output_path="mimic_discharge_summaries.csv",
    )

    # Print first few summaries to verify
    print(df.head())


if __name__ == "__main__":
    main()

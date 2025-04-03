# %%
import re


def chunk_data(data):
    # Define regex pattern for section headers
    pattern = r"(?=\n[A-Za-z\s]+:)"

    # Split data into chunks based on pattern
    chunks = re.split(pattern, data)

    # Create a dictionary to store section headers and corresponding text
    data_dict = {}

    for chunk in chunks:
        # Split each chunk into header and text
        header, *text = chunk.split(":")

        # Join the text back together in case there were multiple colons
        text = ":".join(text)

        # Clean up header and text
        header = header.strip()
        text = text.strip()

        # Store in dictionary
        data_dict[header] = text

    return data_dict


# Load your data
with open("../mimic_discharge_summaries.csv", "r") as file:
    data = file.read()

# Chunk the data
data_dict = chunk_data(data)

# Now you can access the text of each section like this:
# print(data_dict["History of Present Illness"])
print(data_dict["past medical history"])

# %%

import openai
import pandas as pd
from tqdm import tqdm
import time

# Set up your OpenAI API key
openai.api_key = "sk-"

# Load the DataFrame from the Excel sheet
df = pd.read_excel("GPT3.5_ALL_Summaries.xlsx")

# Create a new DataFrame to store the summaries
output_df = pd.DataFrame(columns=['Abstract', 'Claims', 'Abstract Summary', 'Claims Summary', 'New Summary', 'Summary Summary'])

# Retry parameters
max_retries = 3
retry_delay = 5

# Iterate over the rows in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Summaries"):
    # Get the abstract, claims, and existing summaries
    abstract = row['Abstract']
    claims = row['Claims']
    abstract_summary = row['Abstract Summary']
    claims_summary = row['Claims Summary']
    new_summary = row['New Summary']

    # Retry loop for API requests
    for _ in range(max_retries):
        try:
            # Make a request to the OpenAI API for generating the summary of the new summary
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=new_summary,
                max_tokens=50
            )
            break  # Break out of the retry loop if the request is successful
        except openai.error.ServiceUnavailableError:
            time.sleep(retry_delay)

    # Extract the generated summary from the API response
    summary_summary = response.choices[0].text.strip()

    # Append the row to the output DataFrame
    output_df = output_df.append({'Abstract': abstract, 'Claims': claims, 'Abstract Summary': abstract_summary,
                                  'Claims Summary': claims_summary, 'New Summary': new_summary,
                                  'Summary Summary': summary_summary}, ignore_index=True)

# Save the output DataFrame to a new Excel file
output_df.to_excel("GPT3.5_Summary_Summary.xlsx", index=False)

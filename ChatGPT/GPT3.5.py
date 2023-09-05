import openai
import pandas as pd
import re
from tqdm import tqdm

# Set up your OpenAI API key
openai.api_key = "sk-"

# Load the DataFrame from the Excel sheet
df = pd.read_excel("All_Patent.xlsx")

# Create a new DataFrame to store the summaries
output_df = pd.DataFrame(columns=['Abstract', 'Claims', 'Summary'])

# Iterate over the rows in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Summaries"):
    # Get the abstract and claims
    abstract = row['Abstract']
    claims = row['Claims']

    # Clean the abstract and claims text
    abstract = re.sub(r'[^\x00-\x7F]+', '', str(abstract))
    claims = re.sub(r'[^\x00-\x7F]+', '', str(claims))

    # Combine the abstract and claims
    input_text = abstract + ' ' + claims

    # Make a request to the OpenAI API for generating the summary
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates summaries."},
            {"role": "user", "content": input_text}
        ]
    )

    # Extract the generated summary from the API response
    summary = response.choices[0].message.content

    # Append the row to the output DataFrame
    output_df = output_df.append({'Abstract': abstract, 'Claims': claims, 'Summary': summary}, ignore_index=True)

# Save the output DataFrame to a new Excel file
output_df.to_excel("GPT3.5.xlsx", index=False)
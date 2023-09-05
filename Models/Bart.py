import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)

# Load the Excel file
input_file = "All_Patent.xlsx"
df = pd.read_excel(input_file)

# Define a list to store the generated summaries
generated_summaries = []

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

    # Tokenize the input text
    inputs = tokenizer([input_text], max_length=1024, truncation=True, return_tensors='pt').to(device)

    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True).to(device)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    # Add the generated summary to the list
    generated_summaries.append(summary)

# Add the generated summaries to the DataFrame
df['Generated_Summary'] = generated_summaries

# Save the updated DataFrame to a new Excel file
output_file = "Bart_Summary.xlsx"
df.to_excel(output_file, index=False)

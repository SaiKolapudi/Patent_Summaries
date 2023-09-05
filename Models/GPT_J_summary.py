import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-J tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-j-6B')
model = GPT2LMHeadModel.from_pretrained('EleutherAI/gpt-j-6B').to(device)

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

    # Truncate the input text to fit the model's maximum sequence length
    inputs = tokenizer.encode(input_text, truncation=True, max_length=2048, return_tensors='pt').to(device)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=100, num_return_sequences=1, early_stopping=True).to(device)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    # Add the generated summary to the list
    generated_summaries.append(summary)

# Add the generated summaries to the DataFrame
df['Generated_Summary'] = generated_summaries

# Save the updated DataFrame to a new Excel file
output_file = "GPT_J_Summary.xlsx"
df.to_excel(output_file, index=False)

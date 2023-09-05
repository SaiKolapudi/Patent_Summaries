import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)

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
    inputs = tokenizer([input_text], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)

    # Generate the summary
    outputs = model.generate(**inputs)
    summary = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True)

    # Add the generated summary to the list
    generated_summaries.append(summary)

# Add the generated summaries to the DataFrame
df['Generated_Summary'] = generated_summaries

# Save the updated DataFrame to a new Excel file
output_file = "RobertA_Summary.xlsx"
df.to_excel(output_file, index=False)

import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model/tokenizer name or path
model_name_or_path = 'turingmachine/hupd-t5-small'

# Load the Excel file
input_file = "All_Patent.xlsx"
data_frame = pd.read_excel(input_file)

# Initialize an empty list to store the summaries
summaries = []

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
model.to(device)

# Configure the progress bar
progress_bar = tqdm(total=len(data_frame), desc="Processing rows")

# Loop through the rows
for index, row in data_frame.iterrows():
    # Get the abstract and claims
    abstract = row['Abstract']
    claims = row['Claims']

    # Clean the abstract and claims text
    abstract = re.sub(r'[^\x00-\x7F]+', '', abstract)
    claims = re.sub(r'[^\x00-\x7F]+', '', claims)

    # Combine the abstract and claims
    input_text = abstract + ' ' + claims

    # Generate the summary
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1200).to(device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=1200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add the summary to the list
    summaries.append(generated_text)

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Add the summaries to the data frame
data_frame['Summary'] = summaries

# Save the data frame to a new Excel file
output_file = 'hupd-t5-small.xlsx'
data_frame.to_excel(output_file, index=False)
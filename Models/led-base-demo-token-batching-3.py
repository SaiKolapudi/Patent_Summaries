import pandas as pd
import re
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# Load the LED-base model and tokenizer
model_name = "pszemraj/led-base-book-summary"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Load the Excel file
input_file = "All_Patent.xlsx"
data_frame = pd.read_excel(input_file)

# Initialize an empty list to store the summaries
summaries = []

# Create a progress bar
pbar = tqdm(total=len(data_frame))

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
    
    # Encode the combined text to handle special characters
    input_text = input_text.encode('unicode_escape').decode()
    
    # Generate the summary
    summary = summarizer(input_text)[0]['summary_text']
    
    # Add the summary to the list
    summaries.append(summary)
    
    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()

# Add the summaries to the data frame
data_frame['Summary'] = summaries

# Save the data frame to a new Excel file
output_file = 'led-base-demo-token-batching-3.xlsx'
data_frame.to_excel(output_file, index=False)

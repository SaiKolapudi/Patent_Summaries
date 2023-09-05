import pandas as pd
import re
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
from tqdm import tqdm
from rouge_score import rouge_scorer
from rouge_score import scoring
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the Excel file
input_file = "All_Patent.xlsx"
data_frame = pd.read_excel(input_file)

# Initialize an empty list to store the summaries
summaries = []

# Load the LongT5-TGlobal-XL model
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-xl")
model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-xl").to('cuda')

# Loop through the rows with a progress bar
for index, row in tqdm(data_frame.iterrows(), total=len(data_frame), desc="Processing rows"):
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

    # Wrap the input_text inside a tensor and move to GPU
    input_tensor = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to('cuda')

    # Generate the summary
    outputs = model.generate(input_tensor.input_ids)
    
    # Decode the summary from the generated output
    predicted_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add the summary to the list
    summaries.append(predicted_summary)

# Add the summaries to the data frame
data_frame['Summary'] = summaries

# Save the data frame to a new Excel file
output_file = 'Long-T5.xlsx'
data_frame.to_excel(output_file, index=False)
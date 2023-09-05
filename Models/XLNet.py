import re
import pandas as pd
from tqdm import tqdm
from openpyxl import Workbook
from summarizer import TransformerSummarizer
import tensorflow as tf

# Load the DataFrame from the Word document or any other source
df = pd.read_excel("All_Patent.xlsx")

# Create a new DataFrame to store the summaries
output_df = pd.DataFrame(columns=['Abstract', 'Claims', 'Summary'])

# Configure TensorFlow to use GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

    # Summarize the input text using TensorFlow and GPU
    with tf.device('/GPU:0'):
        model = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
        summary = ''.join(model(input_text, min_length=60))

    # Store the summary in the output DataFrame
    output_df = output_df.append({'Abstract': abstract, 'Claims': claims, 'Summary': summary}, ignore_index=True)

# Save the output DataFrame to an Excel file
output_df.to_excel("XLNet_output.xlsx", index=False)

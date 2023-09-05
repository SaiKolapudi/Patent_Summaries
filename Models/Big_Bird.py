import pandas as pd
import re
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
from tqdm import tqdm
from rouge_score import rouge_scorer
from rouge_score import scoring

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the Excel file
input_file = "All_Patent.xlsx"
data_frame = pd.read_excel(input_file)

# Initialize an empty list to store the summaries
summaries = []

# Load the BigBird model
path = 'gs://bigbird-transformer/summarization/pubmed/roberta/saved_model'
imported_model = tf.saved_model.load(path, tags='serve')
summerize = imported_model.signatures['serving_default']

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

    # Wrap the input_text inside a tensor
    input_tensor = tf.constant([input_text])

    # Generate the summary
    predicted_summary = summerize(input_tensor)['pred_sent'][0].numpy().decode('utf-8')

    # Add the summary to the list
    summaries.append(predicted_summary)

# Add the summaries to the data frame
data_frame['Summary'] = summaries

# Save the data frame to a new Excel file
output_file = 'Big_Bird.xlsx'
data_frame.to_excel(output_file, index=False)

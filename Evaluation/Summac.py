### GPT3.5 SummaC_Score
import pandas as pd
from tqdm import tqdm
import torch
import re
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the DataFrame from the Excel file
input_file = "GPT3.5.xlsx"
df = pd.read_excel(input_file)

# Define a list to store SummaC scores
summac_scores = []

# Initialize BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to calculate BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model(**inputs)
    return outputs['last_hidden_state'].mean(dim=1).cpu().detach().numpy()

# Iterate over the rows in the DataFrame with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Scores"):
    # Get the original text and the generated summary
    abstract = row['Abstract']
    claim = row['Claims']

    # Clean the abstract and claims text
    abstract = re.sub(r'[^\x00-\x7F]+', '', str(abstract))
    claims = re.sub(r'[^\x00-\x7F]+', '', str(claim))

    # Combine the cleaned abstract and claims
    original_text = abstract+' '+claims
    generated_summary = str(row['Summary'])

    # Get embeddings for original text and generated summary
    original_text_embedding = get_bert_embeddings(original_text)
    generated_summary_embedding = get_bert_embeddings(generated_summary)

    # Calculate cosine similarity between embeddings
    cos_sim = cosine_similarity(original_text_embedding, generated_summary_embedding)[0][0]

    # Append SummaC score to the list
    summac_scores.append(cos_sim)

# Add the scores to the DataFrame
df['SummaC_Score'] = summac_scores

# Print the average scores
print("Average SummaC Score:", np.mean(summac_scores))

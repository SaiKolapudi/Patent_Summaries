import pandas as pd
import torch
import re
from bert_score import score
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the DataFrame from the Excel file
input_file = "GPT3.5.xlsx"
df = pd.read_excel(input_file)

# Define lists to store the BERT metric scores
bert_metric_scores = []

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a progress bar
progress_bar = tqdm(total=len(df), desc="Calculating BERT Metric Scores")

# Iterate over the rows in the DataFrame
# Iterate over the rows in the DataFrame with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Scores"):
    # Get the original text and the generated summary
    abstract = row['Abstract']
    claims = row['Claims']

    # Combine the cleaned abstract and claims
    original_text = abstract+' '+claims
    generated_summary = str(row['Summary'])

    # Calculate BERT metric scores
    _, _, bert_metric_score = score([generated_summary], [original_text], model_type="bert-base-uncased", device=device)

    # Append the score to the BERT metric scores list
    bert_metric_scores.append(bert_metric_score.item())

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Add the scores to the DataFrame
df["BERT_Metric_Score"] = bert_metric_scores

# Save the updated DataFrame to a new Excel file
# output_file = "Summary_Scores_BERT_metric_t5_small_total.xlsx"
# df.to_excel(output_file, index=False)

# Print the average BERT metric score
print("Average BERT Metric Score:", sum(bert_metric_scores) / len(bert_metric_scores))

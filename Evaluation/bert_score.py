import pandas as pd
import torch
import re
from bert_score import score

# Load the DataFrame from the Excel file
input_file = "XLNet_output.xlsx"
df = pd.read_excel(input_file)

# Define lists to store the BERT metric scores
bert_metric_scores = []

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    # Get the original text and the generated summary
    abstract = row['Abstract']

    # Clean the abstract and claims text
    abstract = re.sub(r'[^\x00-\x7F]+', '', str(abstract))

    # Combine the cleaned abstract and claims
    original_text = abstract
    generated_summary = str(row['Summary'])

    # Calculate BERT metric scores
    _, _, bert_metric_score = score([generated_summary], [original_text], model_type="bert-base-uncased", device=device)

    # Append the score to the BERT metric scores list
    bert_metric_scores.append(bert_metric_score.item())

# Add the scores to the DataFrame
df["BERT_Metric_Score"] = bert_metric_scores

# Save the updated DataFrame to a new Excel file
# output_file = "Summary_Scores_BERT_metric_t5_small_total.xlsx"
# df.to_excel(output_file, index=False)

# Print the average BERT metric score
print("Average BERT Metric Score:", sum(bert_metric_scores) / len(bert_metric_scores))
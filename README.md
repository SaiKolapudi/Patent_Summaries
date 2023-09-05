# Patent Document Summarization

Welcome to the repository for patent document summarization! This codebase encompasses a range of powerful neural network models designed to summarize patent documents, aligning with the insights presented in the research paper titled "Enhancing Summarization Models for Patent Documents".

## Models in Focus
We've meticulously explored and assessed the performance of several cutting-edge models:

- T5 and its intriguing variations (HUPD-T5 Small, HUPD-T5 Base, Long-T5-TGlobal)
- XLNet, an innovative autoregressive method
- BART, famed for its denoising autoencoding prowess
- BigBird, the attention mechanism-revolutionizing transformer
- Pegasus, sought after for its abstractive text generation
- The mighty GPT-3.5, a true natural language processing powerhouse

## Data Unveiled

The dataset comprises 1630 patent documents entrenched in the realm of communication and streaming technology. These gems were meticulously gleaned from the bountiful treasure troves of Google Patents. We've zeroed in on a trio of pivotal data fields:

- Patent numbers, to uniquely identify each patent
- Abstracts, encapsulating the essence of each invention
- Claims, laying down the legal foundation of the patents

Our text preprocessing pipeline is the cornerstone of data refinement, which involves expunging special characters, normalizing whitespace, and ushering text into lowercase serenity.

## Evaluation Spectrum

Diverse facets of the patent documents are put through the evaluation wringer:

- Solely abstracts
- Claims in isolation
- The tandem of abstracts and claims
- Summaries of abstracts alongside summaries of claims
- Summaries of the composite "Summary of Abstract + Summary of Claims"

This multifaceted evaluation journey is underpinned by an array of automated evaluation metrics:

- ROUGE (1, 2, L)
- BLEU
- BERT Score
- Flesch Reading Ease
- Dale Chall Readability
- Coleman-Liau Index
- N-gram
- SummaC

## Glimpses of Triumph

The results section is a treasure trove of insights awaiting your perusal. Key takeaways include:

- HUPD-T5 Small basks in the glory of overall performance supremacy, backed by impressive ROUGE scores
- GPT-3.5 emerges as a versatile contender, flaunting strong results across all metrics
- The amalgamation of full abstracts and claims births summaries of superior quality compared to standalone sections
- Nudging GPT-3.5 with instructions and examples results in a further embellishment of summarization excellence

# Hands-On Implementation

Unlock the power of patent document summarization with seamless integration through Hugging Face Transformers. Our capabilities come alive through the following steps:

1. **Importing Essential Modules**

   Begin by importing the necessary modules for effective summarization:

   ```python
   from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
   ```

2. **Loading the Model and Tokenizer**

   Initialize the model and tokenizer using Hugging Face's convenient methods:

   ```python
   model_name = "HUPD-T5-Small"
   model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

3. **Summarization in Action**

   Effortlessly generate a summary by providing the input text to the model:

   ```python
   def summarize(input_text):
       input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
       summary_ids = model.generate(input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
       summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
       return summary
   ```

4. **Input Text and Summary Generation**

   To kickstart the summarization process, call the `summarize` function:

   ```python
   input_text = "Insert your abstract text here"
   generated_summary = summarize(input_text)
   ```

   Now, `generated_summary` contains the insightful summary of your input text, ready for your exploration.

## Exploration and Beyond

With this streamlined approach, you're empowered to delve into the realms of patent document summarization, courtesy of Hugging Face Transformers. The `summarize.py` module is your gateway to uncovering key insights, and generating impactful summaries has never been more accessible.



## Prerequisites

The key to unlocking this treasure trove requires:

- Python 3.6+
- Transformers
- Seq2Seq
- Rouge
- etc (don't forget to list other essential packages)

## Connect With Us

Should queries beckon, feel free to reach out at saitulasi1729@gmail.com. Your inquiries are valued and anticipated!

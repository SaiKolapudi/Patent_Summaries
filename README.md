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

## Hands-On Interaction

The nucleus of our summarization capabilities is encased within `summarize.py`. To set the wheels in motion:

```python
from summarize import summarize

text = "Insert your abstract text here"
summary = summarize(text, model="HUPD-T5-Small")
```

For result replication, the evaluation scripts reside in `evaluate.py`. Simply execute:

```python
from evaluate import evaluate_all

df = evaluate_all("patent_docs.csv")
```

In this instance, `patent_docs.csv` holds the prized, preprocessed patent text.



## Prerequisites

The key to unlocking this treasure trove requires:

- Python 3.6+
- Transformers
- Seq2Seq
- Rouge
- etc (don't forget to list other essential packages)

## Connect With Us

Should queries beckon, feel free to reach out at junhua.ding@unt.edu. Your inquiries are valued and anticipated!

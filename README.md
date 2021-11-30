# Sequence_Span_Rewriting
Code for EMNLP 2021 paper [Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting](https://aclanthology.org/2021.emnlp-main.45/)


## Usage

`data_generation.py` contains key functions of generating training data for the sequence span rewriting objective.

`data_gen.py` contains an example of data generation.

`run_summarization.py` is from Huggingface Transformers. We use this file to continually per-train with SSR and fine-tune it on downstream tasks.

`run_generation.py` is used for inference (i.e., generation).

## Pre-trained models

You can load our pre-trained `SSR-base` from Huggingface's model hub:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("microsoft/ssr-base")

model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/ssr-base")
```

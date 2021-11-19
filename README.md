# Sequence_Span_Rewriting
Code for EMNLP 2021 paper: Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting


## Usage

data_generation.py contains key functions of generating training data for the sequence span rewriting objective.

data_gen.py contains an example of data generation.

run_summarization.py is from huggingface transformers. We use this function to continually per-train with SSR and fine-tune on downstream tasks.

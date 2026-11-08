---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: summarization
base_model: google/pegasus-large
datasets:
- pariza/bbc-news-summary
tags:
- summarization
- news
- bbc
- newssum
metrics:
- rouge
- bertscore
---

# NewsSum V2: PEGASUS-large fine-tuned on BBC News Summary

PEGASUS-large fine-tuned for news summarisation on the BBC News Summary dataset, as part of **NewsSum V2**, a leakage-free comparison of extractive baselines, fine-tuned transformers and a small LLM. This is the seed-0 model from the study.

- Code, results and Kaggle notebook: [github.com/MHKhan50/NewsSum-Bench](https://github.com/MHKhan50/NewsSum-Bench)
- Paper: *When the Reference Is Extracted: TextRank Outscores Fine-Tuned Transformers and a Small LLM on BBC News Summary* (Zenodo, [doi:10.5281/zenodo.23264727](https://doi.org/10.5281/zenodo.23264727))
- Author: Muzammal Hussain

## Results on the 311 test articles (this checkpoint)

| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore | Words |
|---|---|---|---|---|---|
| 73.8 | 64.8 | 51.1 | 73.1 | 59.2 | 145 |

Across three training seeds the mean ROUGE-Lsum is 72.7 ± 0.5, the best of the seven fine-tuned models in the study.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

name = "{repo}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name)

article = "..."  # article body, without the headline
inputs = tok(article, max_length=1024, truncation=True, return_tensors="pt")
out = model.generate(**inputs, num_beams=4, max_new_tokens=320, no_repeat_ngram_size=3)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Training data

[BBC News Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary) (BBC articles from 2004 and 2005, five topics, one summary per article). Near-duplicate articles (154 of 2,225) were removed before a fixed 70/15/15 split stratified by topic: 1,449 training, 311 validation and 311 test articles. The split is published in the [repository](https://github.com/MHKhan50/NewsSum-Bench/blob/main/results/splits.csv). Headlines were removed from the input.

**Important:** the reference summaries in this dataset are sentences copied from the article (94.8% verbatim) and are about 45% as long as the article. This model therefore learned to *select and copy* sentences in BBC house style, not to write short abstractive summaries.

## Training

Fully fine-tuned with Adafactor (learning rate 2e-4, effective batch 16, linear schedule with 5% warm-up), early stopping on validation loss (6 epochs, best at epoch 6), fp32 on an NVIDIA T4. Input up to 1,024 tokens, output up to 320 tokens. Decoding: beam search with 4 beams, no repeated trigrams. Weights are stored in fp16.

## Limitations

- Trained on about 1,450 BBC articles from 2004 and 2005; style and topics are narrow.
- Scores are ROUGE and BERTScore against one extractive reference per article. They do not measure factual accuracy, and no human evaluation was done.
- On this dataset a length-matched TextRank baseline (no training) scores higher: 82.9 ROUGE-Lsum. See the paper for why.
- The BBC articles are copyrighted by the BBC; this model is for research and education.

## Citation

```bibtex
@misc{hussain2026newssum,
  author = {Hussain, Muzammal},
  title  = {When the Reference Is Extracted: TextRank Outscores Fine-Tuned Transformers and a Small LLM on BBC News Summary},
  year   = {2026},
  doi    = {10.5281/zenodo.23264727},
  url    = {https://github.com/MHKhan50/NewsSum-Bench}
}
```

---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: summarization
base_model: facebook/bart-base
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

# NewsSum V2: BART-base fine-tuned on BBC News Summary

BART-base fine-tuned for news summarisation on the BBC News Summary dataset, as part of **NewsSum V2**, a leakage-free comparison of extractive baselines, fine-tuned transformers and a small LLM. This is the seed-0 model from the study.

- Code, results and Kaggle notebook: [github.com/MHKhan50/NewsSum-Bench](https://github.com/MHKhan50/NewsSum-Bench)
- Paper: *When the Reference Is Extracted: TextRank Outscores Fine-Tuned Transformers and a Small LLM on BBC News Summary* (Zenodo, [doi:10.5281/zenodo.24046039](https://doi.org/10.5281/zenodo.24046039))
- Author: Muzammal Hussain

## Results on the 311 test articles (this checkpoint)

| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore | Words |
|---|---|---|---|---|---|
| 69.3 | 59.7 | 48.4 | 68.4 | 55.1 | 136 |

Across three training seeds the mean ROUGE-Lsum is 67.8 ± 1.3. A quantised ONNX copy in `onnx/` runs in the browser with Transformers.js; it powers the [demo Space](https://huggingface.co/spaces/{user}/newssum-demo).

## Usage

```python
from transformers import pipeline

summarise = pipeline("summarization", model="{repo}")
print(summarise(article, num_beams=4, max_new_tokens=320, no_repeat_ngram_size=3)[0]["summary_text"])
```

In the browser (Transformers.js):

```js
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
const summarise = await pipeline("summarization", "{repo}", { dtype: "q8" });
const [out] = await summarise(article, { max_new_tokens: 320, num_beams: 1 });
```

## Training data

[BBC News Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary) (BBC articles from 2004 and 2005, five topics, one summary per article). Near-duplicate articles (154 of 2,225) were removed before a fixed 70/15/15 split stratified by topic: 1,449 training, 311 validation and 311 test articles. The split is published in the [repository](https://github.com/MHKhan50/NewsSum-Bench/blob/main/results/splits.csv). Headlines were removed from the input.

**Important:** the reference summaries in this dataset are sentences copied from the article (94.8% verbatim) and are about 45% as long as the article. This model therefore learned to *select and copy* sentences in BBC house style, not to write short abstractive summaries.

## Training

Fully fine-tuned with AdamW (learning rate 5e-5, batch 8, linear schedule with 5% warm-up), fp16 autocast on an NVIDIA T4, early stopping on validation loss (8 epochs, best at epoch 6). Input up to 1,024 tokens, output up to 320 tokens. Weights are stored in fp16.

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
  doi    = {10.5281/zenodo.24046039},
  url    = {https://github.com/MHKhan50/NewsSum-Bench}
}
```

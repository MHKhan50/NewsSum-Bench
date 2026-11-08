---
language: en
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
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

# NewsSum V2: Qwen2.5-1.5B-Instruct with LoRA fine-tuned on BBC News Summary

Qwen2.5-1.5B-Instruct with LoRA fine-tuned for news summarisation on the BBC News Summary dataset, as part of **NewsSum V2**, a leakage-free comparison of extractive baselines, fine-tuned transformers and a small LLM. This is the seed-0 model from the study.

- Code, results and Kaggle notebook: [github.com/MHKhan50/NewsSum-Bench](https://github.com/MHKhan50/NewsSum-Bench)
- Paper: *When the Reference Is Extracted: TextRank Outscores Fine-Tuned Transformers and a Small LLM on BBC News Summary* (Zenodo, [doi:10.5281/zenodo.24046039](https://doi.org/10.5281/zenodo.24046039))
- Author: Muzammal Hussain

## Results on the 311 test articles (this checkpoint)

| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore | Words |
|---|---|---|---|---|---|
| 66.7 | 54.6 | 41.8 | 65.5 | 48.6 | 161 |

Across three seeds the mean ROUGE-Lsum is 64.9 ± 0.5. The same model used zero-shot scores 38.3 and writes numbers that are not in the article in 7.8% of cases; after this fine-tuning that falls to 0.8%.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), "{repo}")

n = round(0.455 * len(article.split()))  # length budget used in training
msg = (f"Summarise the following news article in about {{n}} words. Write complete sentences "
       f"and keep the key facts, names and figures. Reply with the summary only.\n\nArticle:\n{{article}}")
prompt = tok.apply_chat_template([{{"role": "user", "content": msg}}], tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
out = model.generate(**ids, max_new_tokens=400, do_sample=False)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training data

[BBC News Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary) (BBC articles from 2004 and 2005, five topics, one summary per article). Near-duplicate articles (154 of 2,225) were removed before a fixed 70/15/15 split stratified by topic: 1,449 training, 311 validation and 311 test articles. The split is published in the [repository](https://github.com/MHKhan50/NewsSum-Bench/blob/main/results/splits.csv). Headlines were removed from the input.

**Important:** the reference summaries in this dataset are sentences copied from the article (94.8% verbatim) and are about 45% as long as the article. This model therefore learned to *select and copy* sentences in BBC house style, not to write short abstractive summaries.

## Training

LoRA adapter (rank 16, alpha 32, dropout 0.05, all linear layers) trained for 3 epochs with AdamW (learning rate 2e-4, effective batch 16), fp16 base weights on an NVIDIA T4, about 70 minutes. Loss only on summary tokens. The prompt asks for about 0.455 × article length in words.

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

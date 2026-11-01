# NewsSum V2: Transformers, a Small LLM and Extractive Baselines on BBC News Summary

[![tests](https://github.com/MHKhan50/news-summarisation/actions/workflows/tests.yml/badge.svg)](https://github.com/MHKhan50/news-summarisation/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23264727.svg)](https://doi.org/10.5281/zenodo.23264727)

**Status:** V2 code complete; experiments running on Kaggle. Results will be
added below when the runs finish.

NewsSum compares ways of summarising news articles on the
[BBC News Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary)
dataset (2,225 BBC articles from 2004 and 2005, five categories, one reference
summary per article). V2 is a rebuild of the first version from scratch, with
a stricter protocol and a wider set of systems: simple extractive baselines,
summarisers used zero-shot, fine-tuned encoder-decoder transformers, and a
small instruction-tuned LLM, both zero-shot and fine-tuned with LoRA.

## Why V2

A review of the first version found problems that make its scores unusable:

1. **Swapped columns.** In the preprocessed CSV, the column named `title` holds
   the full article and `text` holds the reference summary. The models were
   therefore trained in the reverse direction (summary in, article out), and
   ROUGE was computed against the article.
2. **ROUGE arguments reversed** and the test rows chosen by file position, so
   they overlapped the randomly drawn training set.
3. **Duplicates.** The dataset contains repeated articles; with a random split
   copies end up in both training and test.
4. **Model names did not match the code** (for example, "BART-base" was
   `facebook/bart-large-cnn`), and the mT5 run had not converged.

V1 is kept in the git history (tag `v1.0.0`) and in the Zenodo record's first
version. Its results should not be cited.

## What V2 does

| # | Experiment | Question |
|---|---|---|
| 1 | Data audit | How many near-duplicates are there, how much would a naive split leak, and how were the references built? |
| 2 | Extractive baselines | Lead-3, length-matched Lead, length-matched TextRank, and a greedy extractive oracle (upper bound) |
| 3 | Zero-shot | BART-large-CNN and PEGASUS-CNN/DailyMail (trained on another news dataset), Qwen2.5-1.5B-Instruct |
| 4 | Fine-tuning | BART-base, BART-large, T5-base, Flan-T5-base, Flan-T5-large, PEGASUS-large; Qwen2.5-1.5B-Instruct with LoRA; 3 seeds each |
| 5 | Cross-category | BART-base trained on four categories, tested on the fifth |
| 6 | Evaluation | ROUGE-1/2/L/Lsum with 95% bootstrap CIs, paired comparison against Lead, BERTScore, copying statistics and unsupported numbers |

Design choices:

- **Near-duplicates are removed before splitting** (Jaccard similarity of word
  5-grams at least 0.9). The split is 70/15/15, stratified by category, with a
  fixed seed, and is the same for every system (`data/splits.csv` is written by
  the pipeline).
- **The headline is not part of the input.** Each article file starts with its
  headline; models see the body only.
- **Reference spacing is repaired.** The summary files join sentences without a
  space (`...$10.9bn.For the full year...`); this is fixed before scoring so that
  ROUGE-Lsum sees real sentence boundaries.
- **Length is set without looking at the reference.** Length-matched baselines
  and the LLM prompt use `ratio x article length`, where `ratio` is the median
  compression on the training split.
- **Fixed hyperparameters, no tuning on test data.** Models are selected by
  validation loss with early stopping; the test set is decoded once.
- **Scores are reported with uncertainty:** mean and standard deviation over
  three seeds, bootstrap CIs over test articles, and paired bootstrap
  differences against the Lead baseline.

## Results

Pending. The tables in `results/` (`main.csv`, `by_category.csv`,
`cross_category.csv`, `audit_*.csv`) and figures in `results/figures/` will be
added from the Kaggle run.

## Run it

**On Kaggle (recommended).** Import
[`notebooks/newssum_v2_kaggle.ipynb`](notebooks/newssum_v2_kaggle.ipynb) from
this repository (*File → Import Notebook → GitHub*), then:

1. Accelerator **GPU T4 x2**, Internet **On**, and add the dataset
   `pariza/bbc-news-summary`.
2. Run once with `MODE = "smoke"` (about 10 minutes) to check the setup.
3. Set `MODE = "full"` and *Save Version → Save & Run All*.

The full plan needs more than one 12-hour session. Each (model, seed) is a
separate job: finished jobs are stored with their predictions and scores, and
a job that is still running when the time limit approaches saves a checkpoint.
To continue, add the previous version's output to the notebook as input and run
it again; finished jobs are skipped and unfinished ones resume where they
stopped. With two GPUs, two jobs run in parallel.

**Locally** (CPU is enough for the audit, baselines and tests):

```bash
pip install -e ".[dev]"            # audit, baselines, metrics, tests
pip install -e ".[train]"          # + PyTorch, transformers, PEFT, BERTScore
python scripts/run_experiments.py --data "path/to/BBC News Summary" --out outputs --cpu-only
pytest                             # offline tests on synthetic data
NEWSSUM_SMOKE=1 pytest -m smoke    # end-to-end run with tiny models (needs internet)
```

## Repository structure

```
src/newssum/   package: data, audit, baselines, metrics, models, training engine, runner, report
notebooks/     Kaggle notebook
scripts/       command-line entry point
tests/         offline tests (synthetic data in the dataset's folder layout) and a tiny-model smoke test
results/       tables and figures (added after the Kaggle run)
data/          data instructions (the dataset is not redistributed here)
```

## Limitations

- BBC News Summary is small (about 2,100 articles after de-duplication) and
  covers two years of one publisher.
- Its reference summaries are made of sentences taken from the article, so
  ROUGE against them favours copying. The audit and the copying statistics are
  there to make this visible rather than to hide it.
- ROUGE and BERTScore do not measure factual consistency; the unsupported-number
  rate is only a rough check.

## Citation

Hussain, M. (2026). *NewsSum V2: Transformers, a Small LLM and Extractive
Baselines on BBC News Summary*. Zenodo. https://doi.org/10.5281/zenodo.23264727

See also [`CITATION.cff`](CITATION.cff). Please also cite the dataset:

> D. Greene and P. Cunningham. *Practical Solutions to the Problem of Diagonal
> Dominance in Kernel Document Clustering.* Proc. ICML 2006 (the BBC articles),
> and the BBC News Summary dataset on Kaggle (`pariza/bbc-news-summary`), which
> adds an extractive summary to each article.

## Licence

Code: MIT. The BBC articles belong to the BBC and are used here for research
only; they are not included in this repository.

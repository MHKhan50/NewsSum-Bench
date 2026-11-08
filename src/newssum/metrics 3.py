"""Evaluation metrics.

ROUGE uses Google's ``rouge-score`` package (F1, Porter stemmer). ROUGE-Lsum
expects one sentence per line, so both sides are sentence-split first.
Note the argument order: ``score(target=reference, prediction=system)``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .text import lsum_format, numbers, tokens_lower

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


@lru_cache(maxsize=1)
def _scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)


def rouge_per_doc(references: list[str], predictions: list[str]) -> pd.DataFrame:
    """Per-document ROUGE F1 (x100)."""
    sc = _scorer()
    rows = []
    for ref, pred in zip(references, predictions, strict=True):
        s = sc.score(lsum_format(ref), lsum_format(pred or ""))
        rows.append({k: 100 * s[k].fmeasure for k in ROUGE_TYPES})
    return pd.DataFrame(rows)


def extractive_fragments(article_tokens: list[str], summary_tokens: list[str]) -> list[int]:
    """Greedy extractive fragments (Grusky et al., 2018): lengths of the longest
    shared token sequences between summary and article."""
    where: dict[str, list[int]] = {}
    for j, t in enumerate(article_tokens):
        where.setdefault(t, []).append(j)
    frags, i = [], 0
    n_a, n_s = len(article_tokens), len(summary_tokens)
    while i < n_s:
        best = 0
        for j in where.get(summary_tokens[i], ()):
            k = 0
            while i + k < n_s and j + k < n_a and summary_tokens[i + k] == article_tokens[j + k]:
                k += 1
            best = max(best, k)
        if best > 0:
            frags.append(best)
            i += best
        else:
            i += 1
    return frags


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def abstractiveness(article: str, summary: str) -> dict:
    """Coverage, density, novel bigram share, length and unsupported numbers."""
    a, s = tokens_lower(article), tokens_lower(summary or "")
    frags = extractive_fragments(a, s)
    n = max(len(s), 1)
    s_bi = _ngrams(s, 2)
    nums_s = numbers(summary or "")
    return {
        "words": len(s),
        "coverage": sum(frags) / n,
        "density": sum(f * f for f in frags) / n,
        "novel_bigrams": 100 * len(s_bi - _ngrams(a, 2)) / max(len(s_bi), 1),
        "numbers": len(nums_s),
        "unsupported_numbers": len(nums_s - numbers(article)),
    }


def score_predictions(articles: list[str], references: list[str], predictions: list[str]) -> pd.DataFrame:
    """Per-document ROUGE plus abstractiveness statistics."""
    r = rouge_per_doc(references, predictions)
    ab = pd.DataFrame([abstractiveness(a, p) for a, p in zip(articles, predictions)])
    return pd.concat([r, ab], axis=1)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Mean and percentile bootstrap CI over documents."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Mean difference ``a - b`` with 95% CI and the share of resamples where a <= b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "p_le_0": float((means <= 0).mean())}


def bertscore(references: list[str], predictions: list[str], model_type: str = "roberta-large",
              batch_size: int = 32, device: str | None = None) -> np.ndarray:
    """BERTScore F1 (x100, rescaled with baseline). Needs the ``bert-score`` package."""
    from bert_score import score
    _, _, f1 = score([p or "" for p in predictions], references, model_type=model_type, lang="en",
                     rescale_with_baseline=True, batch_size=batch_size, device=device, verbose=False)
    return 100 * f1.numpy()

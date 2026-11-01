"""Extractive baselines and an extractive oracle (upper bound).

All length-matched baselines get the same word budget per article:
``ratio * article_words``, where ``ratio`` is the median compression of the
*training* split. No reference summary is used to set the length.
"""

from __future__ import annotations

import numpy as np

from .text import n_words, split_sentences


def _budget_select(sentences: list[str], order: list[int], budget: int) -> list[int]:
    """Take sentences in ``order`` until the word budget is reached (the sentence
    that crosses the budget is kept only if that lands closer to it)."""
    chosen, total = [], 0
    for i in order:
        w = n_words(sentences[i])
        if total >= budget:
            break
        if total + w > budget and (total + w - budget) > (budget - total) and chosen:
            break
        chosen.append(i)
        total += w
    return sorted(chosen)


def lead_k(article: str, k: int = 3) -> str:
    return " ".join(split_sentences(article)[:k])


def lead_ratio(article: str, ratio: float) -> str:
    sents = split_sentences(article)
    budget = max(1, round(ratio * n_words(article)))
    return " ".join(sents[i] for i in _budget_select(sents, list(range(len(sents))), budget))


def textrank(article: str, ratio: float, damping: float = 0.85, iters: int = 100) -> str:
    """TextRank with TF-IDF cosine similarity between sentences."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sents = split_sentences(article)
    if len(sents) <= 2:
        return " ".join(sents)
    try:
        x = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:  # only stop words
        return lead_ratio(article, ratio)
    sim = (x @ x.T).toarray()
    np.fill_diagonal(sim, 0.0)
    row = sim.sum(axis=1, keepdims=True)
    trans = np.divide(sim, row, out=np.full_like(sim, 1.0 / len(sents)), where=row > 0)
    score = np.full(len(sents), 1.0 / len(sents))
    for _ in range(iters):
        new = (1 - damping) / len(sents) + damping * trans.T @ score
        if np.abs(new - score).sum() < 1e-8:
            score = new
            break
        score = new
    order = list(np.argsort(-score, kind="stable"))
    budget = max(1, round(ratio * n_words(article)))
    return " ".join(sents[i] for i in _budget_select(sents, order, budget))


def oracle(article: str, reference: str, max_sentences: int | None = None) -> str:
    """Greedy extractive oracle: add the sentence that most improves
    ROUGE-1 + ROUGE-2 F1 against the reference; stop when nothing improves."""
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    sents = split_sentences(article)
    chosen: list[int] = []
    best = 0.0
    limit = max_sentences or len(sents)
    while len(chosen) < limit:
        cand, cand_score = None, best
        for i in range(len(sents)):
            if i in chosen:
                continue
            text = " ".join(sents[j] for j in sorted(chosen + [i]))
            s = sc.score(reference, text)
            val = s["rouge1"].fmeasure + s["rouge2"].fmeasure
            if val > cand_score:
                cand, cand_score = i, val
        if cand is None:
            break
        chosen.append(cand)
        best = cand_score
    return " ".join(sents[j] for j in sorted(chosen))


def run_baselines(test_df, ratio: float, include_oracle: bool = True) -> dict[str, list[str]]:
    arts, refs = test_df["article"].tolist(), test_df["summary"].tolist()
    out = {
        "lead-3": [lead_k(a, 3) for a in arts],
        "lead-ratio": [lead_ratio(a, ratio) for a in arts],
        "textrank-ratio": [textrank(a, ratio) for a in arts],
    }
    if include_oracle:
        out["oracle"] = [oracle(a, r) for a, r in zip(arts, refs)]
    return out

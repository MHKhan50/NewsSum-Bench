"""Dataset audit: size, duplicates, leakage under a naive split, and how
extractive (and lead-biased) the reference summaries are."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import random_split_overlap
from .text import normalise, split_sentences


def reference_profile(article: str, summary: str) -> dict:
    """How the reference was built from the article.

    Returns the share of summary sentences that occur verbatim (after
    normalisation) in the article, the relative positions of the matched
    article sentences, and whether the article's first sentence is included.
    """
    art_sents = split_sentences(article)
    art_norm = [normalise(s) for s in art_sents]
    art_all = normalise(article)
    sum_sents = split_sentences(summary)
    matched, positions = 0, []
    for s in sum_sents:
        ns = normalise(s)
        if not ns:
            continue
        if ns in art_all:
            matched += 1
            for i, a in enumerate(art_norm):
                if ns == a or ns in a:
                    positions.append(i / max(len(art_sents) - 1, 1))
                    break
    first = normalise(art_sents[0]) if art_sents else ""
    return {
        "article_sentences": len(art_sents),
        "summary_sentences": len(sum_sents),
        "verbatim_share": matched / max(len(sum_sents), 1),
        "mean_position": float(np.mean(positions)) if positions else np.nan,
        "has_first_sentence": bool(first) and first in normalise(summary),
        "selected_share": len(sum_sents) / max(len(art_sents), 1),
    }


def audit(df: pd.DataFrame, raw: pd.DataFrame, dups: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Run the full audit. ``raw`` is the data before de-duplication."""
    prof = pd.DataFrame([reference_profile(a, s) for a, s in zip(df["article"], df["summary"])])
    prof.insert(0, "doc_id", df["doc_id"].to_numpy())
    prof["category"] = df["category"].to_numpy()
    prof["compression"] = (df["summary_words"] / df["article_words"]).to_numpy()

    overview = pd.DataFrame([
        ("articles in the dataset", len(raw)),
        ("near-duplicate articles removed", len(dups)),
        ("  of which exact duplicates", int(dups["exact"].sum()) if len(dups) else 0),
        ("  of which duplicated across categories", int((~dups["same_category"]).sum()) if len(dups) else 0),
        ("articles used", len(df)),
        ("train / validation / test", " / ".join(str(int((df["split"] == s).sum()))
                                                 for s in ("train", "validation", "test"))),
        ("median article length (words)", int(df["article_words"].median())),
        ("median summary length (words)", int(df["summary_words"].median())),
        ("median compression (summary/article words)", round(float(prof["compression"].median()), 3)),
        ("summary sentences found verbatim in article (%)", round(100 * prof["verbatim_share"].mean(), 1)),
        ("summaries that include the first sentence (%)", round(100 * prof["has_first_sentence"].mean(), 1)),
        ("mean relative position of selected sentences", round(float(prof["mean_position"].mean()), 3)),
    ], columns=["item", "value"])

    by_cat = (df.assign(compression=prof["compression"].to_numpy(),
                        verbatim=prof["verbatim_share"].to_numpy(),
                        first=prof["has_first_sentence"].to_numpy())
              .groupby("category")
              .agg(articles=("doc_id", "size"), article_words=("article_words", "median"),
                   summary_words=("summary_words", "median"), compression=("compression", "median"),
                   verbatim_pct=("verbatim", lambda v: 100 * v.mean()),
                   first_sentence_pct=("first", lambda v: 100 * v.mean()))
              .round(3))

    leak = random_split_overlap(raw, dups, seed=seed)
    leak_summary = pd.DataFrame([{
        "split": "random 85/15 without de-duplication",
        "repeats": len(leak),
        "test_articles": int(leak["n_test"].iat[0]),
        "mean_leaked": round(leak["leaked"].mean(), 1),
        "mean_leaked_pct": round(leak["leaked_pct"].mean(), 2),
    }])

    # Position histogram: where in the article do reference sentences come from?
    pos = []
    for a, s in zip(df["article"], df["summary"]):
        art = split_sentences(a)
        n = len(art)
        summ = normalise(s)
        for i, sent in enumerate(art):
            pos.append({"decile": min(int(10 * i / max(n, 1)), 9), "selected": normalise(sent) in summ})
    position = (pd.DataFrame(pos).groupby("decile")["selected"].mean().mul(100).round(1)
                .rename("selected_pct").reset_index())

    return {"overview": overview, "by_category": by_cat, "duplicates": dups,
            "leakage_random_split": leak_summary, "reference_profile": prof,
            "selection_by_position": position}

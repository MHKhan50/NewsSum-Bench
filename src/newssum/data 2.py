"""Loading, de-duplication and splitting of the BBC News Summary dataset.

Expected layout (Kaggle dataset ``pariza/bbc-news-summary``)::

    .../BBC News Summary/News Articles/<category>/<nnn>.txt
    .../BBC News Summary/Summaries/<category>/<nnn>.txt

Each article file starts with the headline on its first line, followed by the
body. Each summary file holds sentences extracted from the body.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .text import n_words, normalise, repair_bbc_summary

CATEGORIES = ("business", "entertainment", "politics", "sport", "tech")
KAGGLE_ROOTS = ("/kaggle/input",)


def _read(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def find_dataset_root(start: str | Path | None = None) -> Path:
    """Return the folder that contains ``News Articles`` and ``Summaries``."""
    starts = [Path(start)] if start else [Path(p) for p in KAGGLE_ROOTS] + [Path("data")]
    for base in starts:
        if not base.exists():
            continue
        if (base / "News Articles").is_dir() and (base / "Summaries").is_dir():
            return base
        for art in sorted(base.rglob("News Articles")):
            if art.is_dir() and (art.parent / "Summaries").is_dir():
                return art.parent
    raise FileNotFoundError(
        "BBC News Summary not found. On Kaggle add the dataset 'pariza/bbc-news-summary'; "
        "locally pass the folder that contains 'News Articles' and 'Summaries'.")


def load_bbc(root: str | Path | None = None) -> pd.DataFrame:
    """Load all article/summary pairs into a DataFrame.

    Columns: ``doc_id`` (``category/nnn``), ``category``, ``title``, ``article``
    (body without the headline), ``summary`` (sentence spacing repaired),
    ``article_words``, ``summary_words``.
    """
    root = find_dataset_root(root)
    rows = []
    for cat_dir in sorted((root / "News Articles").iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.txt")):
            s_path = root / "Summaries" / cat_dir.name / f.name
            if not s_path.exists():
                continue
            lines = _read(f).strip().splitlines()
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            summary = repair_bbc_summary(_read(s_path))
            rows.append({"doc_id": f"{cat_dir.name}/{f.stem}", "category": cat_dir.name,
                         "title": title, "article": body, "summary": summary})
    if not rows:
        raise FileNotFoundError(f"No article/summary pairs under {root}")
    df = pd.DataFrame(rows)
    df["article_words"] = df["article"].map(n_words)
    df["summary_words"] = df["summary"].map(n_words)
    return df.reset_index(drop=True)


def _shingles(text: str, k: int = 5) -> set[int]:
    toks = normalise(text).split()
    if len(toks) < k:
        return {hash(" ".join(toks))}
    return {hash(" ".join(toks[i:i + k])) for i in range(len(toks) - k + 1)}


def duplicate_groups(df: pd.DataFrame, threshold: float = 0.9, k: int = 5) -> pd.DataFrame:
    """Find exact and near-duplicate articles.

    Two articles are near-duplicates when the Jaccard similarity of their word
    5-gram shingles is at least ``threshold``. Returns one row per duplicate
    article with the ``doc_id`` it duplicates (the first in sorted order), the
    similarity, and whether the normalised texts are identical.
    """
    from scipy import sparse

    ids = df["doc_id"].tolist()
    norm = df["article"].map(normalise).tolist()
    shingle_sets = [_shingles(a, k) for a in df["article"]]
    vocab: dict[int, int] = {}
    rows, cols = [], []
    for i, s in enumerate(shingle_sets):
        for h in s:
            rows.append(i)
            cols.append(vocab.setdefault(h, len(vocab)))
    x = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(ids), len(vocab)))
    inter = (x @ x.T).tocoo()
    sizes = np.asarray(x.sum(axis=1)).ravel()

    parent = list(range(len(ids)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sim_of: dict[tuple[int, int], float] = {}
    for i, j, v in zip(inter.row, inter.col, inter.data):
        if i >= j:
            continue
        jac = v / (sizes[i] + sizes[j] - v)
        if jac >= threshold:
            sim_of[(i, j)] = float(jac)
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    out = []
    for i in range(len(ids)):
        r = find(i)
        if r != i:
            key = (min(i, r), max(i, r))
            out.append({"doc_id": ids[i], "duplicate_of": ids[r],
                        "jaccard": round(sim_of.get(key, threshold), 4),
                        "exact": norm[i] == norm[r],
                        "same_category": df["category"].iat[i] == df["category"].iat[r]})
    return pd.DataFrame(out, columns=["doc_id", "duplicate_of", "jaccard", "exact", "same_category"])


def deduplicate(df: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop near-duplicates, keeping the first article of each group."""
    dups = duplicate_groups(df, threshold)
    keep = ~df["doc_id"].isin(dups["doc_id"])
    return df[keep].reset_index(drop=True), dups


def make_splits(df: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = 42) -> pd.Series:
    """Stratified (by category) train/validation/test assignment, fixed seed."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, name="split")
    for _, idx in df.groupby("category").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test))
        n_val = int(round(len(idx) * val))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "validation"
    return split


def random_split_overlap(df: pd.DataFrame, dups: pd.DataFrame, test: float = 0.15,
                         n_repeats: int = 20, seed: int = 0) -> pd.DataFrame:
    """Share of test articles with a (near-)duplicate in training under a naive
    random split of the *un-deduplicated* data."""
    group = {d: d for d in df["doc_id"]}
    for a, b in zip(dups["doc_id"], dups["duplicate_of"]):
        group[a] = b
    g = df["doc_id"].map(group).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats):
        perm = rng.permutation(len(df))
        n_test = int(round(len(df) * test))
        te, tr = perm[:n_test], perm[n_test:]
        tr_groups = set(g[tr])
        leaked = np.array([g[i] in tr_groups for i in te])
        rows.append({"repeat": r, "n_test": n_test, "leaked": int(leaked.sum()),
                     "leaked_pct": 100 * leaked.mean()})
    return pd.DataFrame(rows)


def compression_ratio(train: pd.DataFrame) -> float:
    """Median summary/article length ratio in words, estimated on training data only."""
    return float((train["summary_words"] / train["article_words"]).median())


def prepare(root: str | Path | None = None, threshold: float = 0.9, seed: int = 42):
    """Load, de-duplicate and split. Returns ``(df, dups, raw)``; ``df`` has a ``split`` column."""
    raw = load_bbc(root)
    df, dups = deduplicate(raw, threshold)
    df["split"] = make_splits(df, seed=seed)
    return df, dups, raw

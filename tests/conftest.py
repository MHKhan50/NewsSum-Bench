"""Synthetic data in the exact BBC News Summary folder layout, for offline tests."""

import numpy as np
import pytest

from newssum.data import CATEGORIES

WORDS = ("government company market players film music minister growth shares club season "
         "technology users phone election party profits album award coach league network "
         "said would could new year people first last report sales rose fell million percent").split()


def _sentence(rng, n=None):
    n = n or int(rng.integers(8, 20))
    w = [str(rng.choice(WORDS)) for _ in range(n)]
    if rng.random() < 0.3:
        w.insert(int(rng.integers(1, n)), f"{int(rng.integers(2, 99))}%")
    w[0] = w[0].capitalize()
    return " ".join(w) + "."


def make_bbc(root, per_cat=30, seed=0):
    rng = np.random.default_rng(seed)
    base = root / "BBC News Summary"
    first_article = None
    for cat in CATEGORIES:
        (base / "News Articles" / cat).mkdir(parents=True)
        (base / "Summaries" / cat).mkdir(parents=True)
        for i in range(1, per_cat + 1):
            sents = [_sentence(rng) for _ in range(int(rng.integers(8, 16)))]
            title = " ".join(str(rng.choice(WORDS)) for _ in range(5)).capitalize()
            paras = "\n\n".join(" ".join(sents[j:j + 2]) for j in range(0, len(sents), 2))
            pick = sorted({0, 1, *rng.choice(len(sents), size=3, replace=False).tolist()})
            summary = "".join(sents[j] for j in pick)  # no spaces, as in the real files
            article = f"{title}\n\n{paras}\n"
            if first_article is None:
                first_article = (article, summary)
            (base / "News Articles" / cat / f"{i:03d}.txt").write_text(article, encoding="utf-8")
            (base / "Summaries" / cat / f"{i:03d}.txt").write_text(summary, encoding="utf-8")
    # Duplicates: an exact copy in the same category, a near copy, and a cross-category copy.
    art, summ = first_article
    for cat, name, text in [(CATEGORIES[0], "901", art), (CATEGORIES[0], "902", art.replace(".", ".", 1) + "Extra."),
                            (CATEGORIES[3], "903", art)]:
        (base / "News Articles" / cat / f"{name}.txt").write_text(text, encoding="utf-8")
        (base / "Summaries" / cat / f"{name}.txt").write_text(summ, encoding="utf-8")
    # A cp1252-encoded file, as found in the real data.
    (base / "News Articles" / CATEGORIES[1] / "904.txt").write_bytes(
        "Pound note\n\nIt cost £5 at the time. Sales rose 3% in the year to March.\n".encode("cp1252"))
    (base / "Summaries" / CATEGORIES[1] / "904.txt").write_bytes("It cost £5 at the time.".encode("cp1252"))
    return base


@pytest.fixture(scope="session")
def bbc_root(tmp_path_factory):
    return make_bbc(tmp_path_factory.mktemp("bbc"))

import pandas as pd

from newssum.data import (deduplicate, find_dataset_root, load_bbc, make_splits,
                          random_split_overlap)


def test_load_layout_and_encoding(bbc_root):
    assert find_dataset_root(bbc_root.parent) == bbc_root
    df = load_bbc(bbc_root)
    assert len(df) == 5 * 30 + 4
    row = df[df["doc_id"] == "entertainment/904"].iloc[0]
    assert row["title"] == "Pound note"
    assert "£5" in row["article"] and not row["article"].startswith("Pound")
    # sentence spacing repaired in references
    assert ".T" not in df["summary"].iat[0].replace("U.S", "")


def test_duplicates_found_and_removed(bbc_root):
    df = load_bbc(bbc_root)
    clean, dups = deduplicate(df)
    assert set(dups["doc_id"]) == {"business/901", "business/902", "sport/903"}
    assert set(dups["duplicate_of"]) == {"business/001"}
    assert not dups.set_index("doc_id").loc["sport/903", "same_category"]
    assert len(clean) == len(df) - 3


def test_splits_stratified_and_deterministic(bbc_root):
    clean, _ = deduplicate(load_bbc(bbc_root))
    a, b = make_splits(clean, seed=42), make_splits(clean, seed=42)
    assert a.equals(b)
    counts = pd.crosstab(clean["category"], a)
    assert (counts["test"] >= 4).all() and (counts["validation"] >= 4).all()


def test_naive_split_leaks_duplicates(bbc_root):
    df = load_bbc(bbc_root)
    _, dups = deduplicate(df)
    leak = random_split_overlap(df, dups, n_repeats=50)
    assert leak["leaked"].sum() > 0

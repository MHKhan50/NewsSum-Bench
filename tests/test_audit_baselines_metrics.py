import numpy as np

from newssum.audit import audit, reference_profile
from newssum.baselines import lead_k, lead_ratio, oracle, textrank
from newssum.data import prepare
from newssum.metrics import abstractiveness, bootstrap_ci, paired_bootstrap, rouge_per_doc
from newssum.text import n_words

ART = ("The minister announced a new plan. Sales rose 5% last year. The company said profits fell. "
       "Analysts expect growth. Shares closed higher on Friday. The plan starts in May.")


def test_reference_profile_extractive():
    p = reference_profile(ART, "The minister announced a new plan. Analysts expect growth.")
    assert p["verbatim_share"] == 1.0 and p["has_first_sentence"]
    assert p["summary_sentences"] == 2


def test_audit_tables(bbc_root):
    df, dups, raw = prepare(bbc_root.parent)
    t = audit(df, raw, dups)
    ov = dict(zip(t["overview"]["item"], t["overview"]["value"]))
    assert ov["near-duplicate articles removed"] == 3
    assert ov["summary sentences found verbatim in article (%)"] > 95
    assert ov["summaries that include the first sentence (%)"] > 95
    assert set(t["by_category"].index) == {"business", "entertainment", "politics", "sport", "tech"}


def test_baselines_respect_budget():
    assert lead_k(ART, 2) == "The minister announced a new plan. Sales rose 5% last year."
    out = lead_ratio(ART, 0.4)
    assert out.startswith("The minister") and abs(n_words(out) - 0.4 * n_words(ART)) <= 6
    tr = textrank(ART, 0.4)
    assert 0 < n_words(tr) <= n_words(ART)


def test_oracle_beats_lead():
    ref = "Analysts expect growth. The plan starts in May."
    orc = oracle(ART, ref)
    r = rouge_per_doc([ref, ref], [orc, lead_k(ART, 2)])
    assert r["rouge2"].iat[0] == 100.0 and r["rouge2"].iat[0] > r["rouge2"].iat[1]


def test_rouge_argument_order():
    # precision != recall: a long prediction containing the reference has high recall, low F1
    ref = "Sales rose 5% last year."
    r = rouge_per_doc([ref], [ART])
    assert 0 < r["rouge1"].iat[0] < 50


def test_abstractiveness():
    a = abstractiveness(ART, "Sales rose 5% last year. Profits hit 9% in 2004.")
    assert a["unsupported_numbers"] == 2  # 9 and 2004
    assert a["numbers"] == 3
    copy = abstractiveness(ART, "The minister announced a new plan.")
    assert copy["coverage"] == 1.0 and copy["novel_bigrams"] == 0


def test_bootstrap():
    rng = np.random.default_rng(0)
    m, lo, hi = bootstrap_ci(rng.normal(10, 1, 200))
    assert lo < m < hi and hi - lo < 0.5
    pb = paired_bootstrap(np.ones(50) * 2, np.ones(50))
    assert pb["diff"] == 1 and pb["p_le_0"] == 0

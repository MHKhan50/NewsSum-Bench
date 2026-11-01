"""Steps that run in the main (CPU) process: data preparation, audit, baselines
and BERTScore. GPU training runs in worker processes (see ``runner``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import baselines
from .audit import audit
from .data import prepare
from .metrics import bertscore, score_predictions
from .runner import Job, job_frames, load_prepared, save_prepared


def prepare_data(out: str | Path, root: str | Path | None = None, threshold: float = 0.9,
                 seed: int = 42) -> dict[str, pd.DataFrame]:
    """Load, de-duplicate, split and audit. Writes ``data/`` and ``results/audit_*.csv``."""
    out = Path(out)
    df, dups, raw = prepare(root, threshold, seed)
    meta = save_prepared(out, df, {"raw": len(raw), "removed_duplicates": len(dups),
                                   "split_seed": seed, "dedup_threshold": threshold})
    tables = audit(df, raw, dups)
    res = out / "results"
    res.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(res / f"audit_{name}.csv", index=name == "by_category")
    print(f"{meta['raw']} articles, {meta['removed_duplicates']} near-duplicates removed, "
          f"{meta['n']} used; training compression ratio {meta['ratio']:.3f}")
    return tables


def run_baselines(out: str | Path, smoke: bool = False, include_oracle: bool = True) -> pd.DataFrame:
    """Extractive baselines and the oracle on the test split, stored like model runs."""
    out = Path(out)
    df, meta = load_prepared(out)
    te = job_frames(df, Job("baseline"), smoke)[2]
    rows = []
    t0 = time.time()
    for name, preds in baselines.run_baselines(te, meta["ratio"], include_oracle).items():
        d = out / "runs" / Job(name).name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"doc_id": te["doc_id"], "prediction": preds}).to_json(
            d / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
        sc = score_predictions(te["article"].tolist(), te["summary"].tolist(), preds)
        sc.insert(0, "doc_id", te["doc_id"].to_numpy())
        sc.insert(1, "category", te["category"].to_numpy())
        sc.to_csv(d / "scores.csv", index=False)
        mean = sc.drop(columns=["doc_id", "category"]).mean().round(4).to_dict()
        (d / "result.json").write_text(json.dumps({"job": Job(name).name, "model": name, "setting": "main",
                                                   "seed": 0, "n_test": len(te), "ratio": meta["ratio"],
                                                   "mean": mean, "smoke": smoke}, indent=2))
        rows.append({"system": name, **{k: mean[k] for k in ("rouge1", "rouge2", "rougeLsum", "words")}})
    print(f"baselines done in {(time.time() - t0) / 60:.1f} min")
    return pd.DataFrame(rows).round(2)


def add_bertscore(out: str | Path, model_type: str = "roberta-large", batch_size: int = 32) -> int:
    """Add a ``bertscore`` column to every finished run that lacks one."""
    out = Path(out)
    df, _ = load_prepared(out)
    ref = df.set_index("doc_id")["summary"]
    n = 0
    for d in sorted((out / "runs").glob("*")):
        sc_path = d / "scores.csv"
        if not (d / "result.json").exists() or not sc_path.exists():
            continue
        sc = pd.read_csv(sc_path, dtype={"doc_id": str})
        if "bertscore" in sc.columns:
            continue
        pr = pd.read_json(d / "predictions.jsonl", lines=True, dtype={"doc_id": str}).set_index("doc_id")
        preds = pr.loc[sc["doc_id"], "prediction"].fillna("").tolist()
        sc["bertscore"] = bertscore(ref.loc[sc["doc_id"]].tolist(), preds, model_type, batch_size)
        sc.to_csv(sc_path, index=False)
        n += 1
        print(f"BERTScore {d.name}: {sc['bertscore'].mean():.2f}")
    return n

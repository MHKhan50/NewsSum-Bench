"""Result tables and figures from the finished runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci, paired_bootstrap
from .models import REGISTRY

BASELINES = {"lead-3": "extractive baseline", "lead-ratio": "extractive baseline",
             "textrank-ratio": "extractive baseline", "oracle": "extractive oracle (upper bound)"}
METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore", "words", "coverage", "density",
           "novel_bigrams"]


def system_type(name: str) -> str:
    if name in BASELINES:
        return BASELINES[name]
    spec = REGISTRY.get(name)
    if spec is None:
        return "other"
    if not spec.finetune:
        return "zero-shot"
    return "fine-tuned (LoRA)" if spec.lora else "fine-tuned"


def collect(out: str | Path, include_smoke: bool = False) -> pd.DataFrame:
    """All per-document scores of finished runs in one long table."""
    frames = []
    for d in sorted((Path(out) / "runs").glob("*")):
        rp, sp = d / "result.json", d / "scores.csv"
        if not (rp.exists() and sp.exists()):
            continue
        r = json.loads(rp.read_text())
        if r.get("smoke") and not include_smoke:
            continue
        sc = pd.read_csv(sp, dtype={"doc_id": str})
        sc.insert(0, "system", r["model"])
        sc.insert(1, "setting", r["setting"])
        sc.insert(2, "seed", r["seed"])
        frames.append(sc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main_table(per_doc: pd.DataFrame, baseline: str = "lead-ratio", n_boot: int = 1000,
               seed: int = 0) -> pd.DataFrame:
    """One row per system on the in-domain test split.

    ROUGE values are means over test documents, averaged over seeds; ``*_sd`` is
    the standard deviation across seeds; ``rougeLsum_lo/hi`` is a 95% bootstrap
    CI over documents (seed-averaged scores). ``vs_*`` columns compare with the
    length-matched Lead baseline by paired bootstrap.
    """
    main = per_doc[per_doc["setting"] == "main"]
    cols = [c for c in METRICS + ["unsupported_numbers", "numbers"] if c in main.columns]
    doc = main.groupby(["system", "doc_id"])[cols].mean()
    by_seed = main.groupby(["system", "seed"])[[c for c in ("rouge1", "rouge2", "rougeLsum") if c in cols]].mean()
    base = doc.loc[baseline] if baseline in doc.index.get_level_values(0) else None
    rows = []
    for system, g in doc.groupby(level=0):
        g = g.droplevel(0)
        row = {"system": system, "type": system_type(system),
               "seeds": int(main.loc[main["system"] == system, "seed"].nunique()), "n": len(g)}
        for c in cols:
            if c not in ("unsupported_numbers", "numbers"):
                row[c] = g[c].mean()
        s = by_seed.loc[system]
        for c in s.columns:
            row[f"{c}_sd"] = s[c].std(ddof=1) if len(s) > 1 else np.nan
        _, row["rougeLsum_lo"], row["rougeLsum_hi"] = bootstrap_ci(g["rougeLsum"].to_numpy(), n_boot, seed)
        if "numbers" in g:
            row["unsupported_numbers_pct"] = 100 * g["unsupported_numbers"].sum() / max(g["numbers"].sum(), 1)
        if base is not None and system != baseline:
            b = base.loc[g.index]
            for m in ("rouge2", "rougeLsum"):
                pb = paired_bootstrap(g[m].to_numpy(), b[m].to_numpy(), n_boot, seed)
                row[f"vs_{baseline}_{m}"] = pb["diff"]
                row[f"vs_{baseline}_{m}_lo"] = pb["lo"]
                row[f"vs_{baseline}_{m}_hi"] = pb["hi"]
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("rougeLsum", ascending=False)
    return t.round(3).reset_index(drop=True)


def category_table(per_doc: pd.DataFrame, metric: str = "rougeLsum") -> pd.DataFrame:
    main = per_doc[per_doc["setting"] == "main"]
    t = main.groupby(["system", "category"])[metric].mean().unstack()
    t["all"] = main.groupby("system")[metric].mean()
    return t.sort_values("all", ascending=False).round(2)


def cross_category_table(per_doc: pd.DataFrame, model: str = "bart-base") -> pd.DataFrame:
    """Held-out-category score vs the same model trained on all categories (seed 0),
    on exactly the same test documents."""
    rows = []
    x = per_doc[(per_doc["system"] == model) & per_doc["setting"].str.startswith("xcat-")]
    ind = per_doc[(per_doc["system"] == model) & (per_doc["setting"] == "main") & (per_doc["seed"] == 0)]
    ind = ind.set_index("doc_id")
    for setting, g in x.groupby("setting"):
        g = g.set_index("doc_id")
        common = g.index.intersection(ind.index)
        if len(common) == 0:
            continue
        row = {"held_out": setting.split("-", 1)[1], "n": len(common)}
        for m in ("rouge2", "rougeLsum"):
            row[f"{m}_in_domain"] = ind.loc[common, m].mean()
            row[f"{m}_held_out"] = g.loc[common, m].mean()
            pb = paired_bootstrap(g.loc[common, m].to_numpy(), ind.loc[common, m].to_numpy())
            row[f"{m}_diff"], row[f"{m}_lo"], row[f"{m}_hi"] = pb["diff"], pb["lo"], pb["hi"]
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def training_table(out: str | Path) -> pd.DataFrame:
    rows = []
    for rp in sorted((Path(out) / "runs").glob("*/result.json")):
        r = json.loads(rp.read_text())
        if not r.get("history") or r.get("smoke"):
            continue
        best = min(r["history"], key=lambda h: h["val_loss"])
        rows.append({"job": r["job"], "epochs_run": len(r["history"]), "best_epoch": best["epoch"],
                     "best_val_loss": round(best["val_loss"], 4),
                     "minutes_train": r.get("minutes_train"), "minutes_generate": r.get("minutes_generate"),
                     "gpu": r.get("versions", {}).get("gpu")})
    return pd.DataFrame(rows)


def write_all(out: str | Path, n_boot: int = 1000) -> dict[str, pd.DataFrame]:
    out = Path(out)
    res = out / "results"
    (res / "figures").mkdir(parents=True, exist_ok=True)
    per_doc = collect(out)
    if per_doc.empty:
        print("no finished runs yet")
        return {}
    tables = {"main": main_table(per_doc, n_boot=n_boot), "by_category": category_table(per_doc),
              "cross_category": cross_category_table(per_doc), "training": training_table(out)}
    per_doc.to_csv(res / "per_document_scores.csv.gz", index=False)
    for k, t in tables.items():
        t.to_csv(res / f"{k}.csv", index=k == "by_category")
    figures(tables["main"], res / "figures", out)
    return tables


def figures(main: pd.DataFrame, fig_dir: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"extractive baseline": "#8c8c8c", "extractive oracle (upper bound)": "#c9c9c9",
               "zero-shot": "#e69f00", "fine-tuned": "#0072b2", "fine-tuned (LoRA)": "#009e73"}
    t = main.sort_values("rougeLsum")
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(t) + 1))
    err = np.vstack([t["rougeLsum"] - t["rougeLsum_lo"], t["rougeLsum_hi"] - t["rougeLsum"]])
    ax.barh(t["system"], t["rougeLsum"], xerr=err, color=[colours.get(x, "#555") for x in t["type"]],
            capsize=2)
    ax.set_xlabel("ROUGE-Lsum F1 (95% bootstrap CI)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for k, c in colours.items() if k in set(t["type"])]
    ax.legend(handles, [k for k in colours if k in set(t["type"])], loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "rouge_lsum_by_system.png", dpi=200)
    plt.close(fig)

    if "density" in t:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for typ, g in t.groupby("type"):
            ax.scatter(g["density"], g["rouge2"], color=colours.get(typ, "#555"), label=typ)
            for _, r in g.iterrows():
                ax.annotate(r["system"], (r["density"], r["rouge2"]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_xlabel("extractive density (higher = more copying)")
        ax.set_ylabel("ROUGE-2 F1")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "copying_vs_rouge2.png", dpi=200)
        plt.close(fig)

    pos_path = out / "results" / "audit_selection_by_position.csv"
    if pos_path.exists():
        pos = pd.read_csv(pos_path)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(pos["decile"] * 10 + 5, pos["selected_pct"], width=9, color="#0072b2")
        ax.set_xlabel("position of sentence in article (%)")
        ax.set_ylabel("sentences in reference (%)")
        fig.tight_layout()
        fig.savefig(fig_dir / "reference_position.png", dpi=200)
        plt.close(fig)

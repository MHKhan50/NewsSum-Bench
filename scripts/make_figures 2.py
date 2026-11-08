"""Publication figures from results/main.csv and the audit tables (PNG and PDF)."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "figures"
NAMES = {"oracle": "Extractive oracle (upper bound)", "textrank-ratio": "TextRank, length-matched",
         "pegasus-large": "PEGASUS-large", "bart-large": "BART-large", "bart-base": "BART-base",
         "t5-base": "T5-base", "qwen2.5-1.5b-lora": "Qwen2.5-1.5B + LoRA", "flan-t5-base": "Flan-T5-base",
         "flan-t5-large": "Flan-T5-large", "lead-ratio": "Lead, length-matched", "lead-3": "Lead-3",
         "qwen2.5-1.5b-zs": "Qwen2.5-1.5B (zero-shot)", "bart-large-cnn": "BART-large-CNN (zero-shot)",
         "pegasus-cnn": "PEGASUS-CNN (zero-shot)"}
COL = {"extractive baseline": "#8a94a6", "extractive oracle (upper bound)": "#c9ced8", "zero-shot": "#e69f00",
       "fine-tuned": "#1f4fd1", "fine-tuned (LoRA)": "#0f9d7a"}
LABEL = {"extractive baseline": "Extractive baseline", "extractive oracle (upper bound)": "Extractive oracle",
         "zero-shot": "Zero-shot", "fine-tuned": "Fine-tuned", "fine-tuned (LoRA)": "Fine-tuned with LoRA"}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RES / "main.csv").sort_values("rougeLsum")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    err = np.vstack([m.rougeLsum - m.rougeLsum_lo, m.rougeLsum_hi - m.rougeLsum])
    ax.barh([NAMES[s] for s in m.system], m.rougeLsum, xerr=err, color=[COL[t] for t in m.type],
            capsize=2, height=0.7)
    for y, (v, hi) in enumerate(zip(m.rougeLsum, m.rougeLsum_hi)):
        ax.text(hi + 1.2, y, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel("ROUGE-Lsum F1 on 311 test articles (95% bootstrap CI)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()], LABEL.values(),
              loc="lower right", fontsize=7.5, frameon=False)
    save(fig, "rouge_lsum_by_system")

    p = pd.read_csv(RES / "audit_selection_by_position.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(p.decile * 10 + 5, p.selected_pct, width=8.6, color="#1f4fd1")
    for x, v in zip(p.decile * 10 + 5, p.selected_pct):
        ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{i}%" for i in range(0, 101, 10)])
    ax.set_xlabel("Position of the sentence in the article")
    ax.set_ylabel("Copied into the\nreference (%)")
    ax.set_ylim(0, 72)
    save(fig, "reference_position")

    t = m[m.system != "oracle"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for typ, g in t.groupby("type"):
        ax.scatter(g.words, g.rougeLsum, color=COL[typ], label=LABEL[typ], s=36, zorder=3)
    offsets = {"bart-base": (4, -9), "t5-base": (4, 3), "flan-t5-base": (4, -9), "qwen2.5-1.5b-lora": (4, 3),
               "lead-ratio": (-8, -12), "textrank-ratio": (-8, 6)}
    for _, r in t.iterrows():
        dx, dy = offsets.get(r.system, (4, 3))
        ha = "right" if dx < 0 else "left"
        ax.annotate(NAMES[r.system].replace(" (zero-shot)", ""), (r.words, r.rougeLsum), fontsize=7,
                    xytext=(dx, dy), textcoords="offset points", ha=ha)
    ax.axvline(178, color="#999", lw=0.8, ls="--")
    ax.text(176, 22, "length budget of the\nmatched baselines", fontsize=7, ha="right", color="#666")
    ax.set_xlabel("Mean summary length (words)")
    ax.set_ylabel("ROUGE-Lsum F1")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "length_vs_rouge")


if __name__ == "__main__":
    main()

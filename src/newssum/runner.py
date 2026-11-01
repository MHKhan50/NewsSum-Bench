"""Experiment runner: job plan, resumable execution and multi-GPU workers.

Every (model, setting, seed) is one job with its own folder under
``<out>/runs/<job>``. A job is finished when ``result.json`` exists, so a new
Kaggle session simply skips finished jobs and resumes unfinished ones from their
checkpoint. With two GPUs (Kaggle T4 x2), two worker processes take jobs from
the same list, one per GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from . import __version__
from .data import CATEGORIES, compression_ratio
from .models import MAIN_ORDER, XCAT_MODEL, get_spec
from .text import n_words


@dataclass(frozen=True)
class Job:
    model: str
    setting: str = "main"         # "main" or "xcat-<category>" (held-out category)
    seed: int = 0

    @property
    def name(self) -> str:
        return f"{self.model}__{self.setting}__s{self.seed}"


def plan(mode: str = "full", seeds: int = 3, models: list[str] | None = None,
         cross_category: bool = True) -> list[Job]:
    """Job list in execution order (cheap first)."""
    if mode == "smoke":
        return [Job(m, "main", 0) for m in (models or ["bart-large-cnn", "bart-base", "flan-t5-base",
                                                       "qwen2.5-1.5b-lora"])]
    jobs = []
    for m in models or MAIN_ORDER:
        n = seeds if get_spec(m).finetune else 1
        jobs += [Job(m, "main", s) for s in range(n)]
        if cross_category and m == XCAT_MODEL:
            jobs += [Job(m, f"xcat-{c}", 0) for c in CATEGORIES]
    return jobs


SMOKE_OVERRIDES = {"epochs": 1, "max_target": 64, "est_minutes": 3}
SMOKE_SIZES = {"train": 48, "validation": 12, "test": 12}


# ---------------------------------------------------------------- files and claims

def run_dir(out: Path, job: Job) -> Path:
    return out / "runs" / job.name


def is_done(out: Path, job: Job) -> bool:
    return (run_dir(out, job) / "result.json").exists()


def claim(out: Path, job: Job, worker: str) -> bool:
    d = run_dir(out, job)
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, worker.encode())
    os.close(fd)
    return True


def release(out: Path, job: Job) -> None:
    (run_dir(out, job) / ".claim").unlink(missing_ok=True)


def clear_stale_claims(out: Path) -> int:
    n = 0
    for c in (out / "runs").glob("*/.claim"):
        c.unlink()
        n += 1
    return n


def status(out: Path, jobs: list[Job]) -> pd.DataFrame:
    rows = []
    for j in jobs:
        d = run_dir(out, j)
        state = "done" if is_done(out, j) else ("partial" if (d / "ckpt" / "state.pt").exists()
                                                 else ("failed" if (d / "error.txt").exists() else "pending"))
        rows.append({"job": j.name, "model": j.model, "setting": j.setting, "seed": j.seed, "status": state})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- data for a job

def load_prepared(out: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_json(out / "data" / "dataset.jsonl", lines=True, dtype={"doc_id": str})
    meta = json.loads((out / "data" / "meta.json").read_text())
    return df, meta


def save_prepared(out: Path, df: pd.DataFrame, extra: dict | None = None) -> dict:
    (out / "data").mkdir(parents=True, exist_ok=True)
    df.to_json(out / "data" / "dataset.jsonl", orient="records", lines=True, force_ascii=False)
    df[["doc_id", "category", "split"]].to_csv(out / "data" / "splits.csv", index=False)
    meta = {"ratio": compression_ratio(df[df["split"] == "train"]), "n": len(df), **(extra or {})}
    (out / "data" / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def job_frames(df: pd.DataFrame, job: Job, smoke: bool = False):
    if job.setting == "main":
        tr, va, te = (df[df["split"] == s] for s in ("train", "validation", "test"))
    else:
        held = job.setting.split("-", 1)[1]
        inside = df["category"] != held
        tr = df[inside & (df["split"] == "train")]
        va = df[inside & (df["split"] == "validation")]
        te = df[~inside & (df["split"] == "test")]
    if smoke:
        tr, va, te = (x.head(SMOKE_SIZES[k]) for x, k in ((tr, "train"), (va, "validation"), (te, "test")))
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def target_words(articles, ratio: float) -> list[int]:
    return [max(10, round(ratio * n_words(a))) for a in articles]


# ---------------------------------------------------------------- one job

def run_job(out: Path, job: Job, deadline: float, smoke: bool = False, save_models: bool = True,
            hf_overrides: dict | None = None, log=print) -> None:
    from . import engine
    from .metrics import score_predictions

    df, meta = load_prepared(out)
    overrides = dict(SMOKE_OVERRIDES) if smoke else {}
    if hf_overrides and job.model in hf_overrides:
        overrides["hf_id"] = hf_overrides[job.model]
    spec = get_spec(job.model, **overrides)
    d = run_dir(out, job)
    d.mkdir(parents=True, exist_ok=True)
    tr, va, te = job_frames(df, job, smoke)
    ratio = meta["ratio"]
    t0 = time.time()
    engine.set_seed(job.seed)
    log(f"[{job.name}] {spec.hf_id} train={len(tr)} val={len(va)} test={len(te)}")
    tok, model = engine.load(spec, train=spec.finetune)
    spec = engine.fit_lengths(spec, model)
    history = []
    if spec.finetune:
        tr_data = engine.encode(spec, tok, tr["article"].tolist(), tr["summary"].tolist(),
                                target_words(tr["article"], ratio))
        va_data = engine.encode(spec, tok, va["article"].tolist(), va["summary"].tolist(),
                                target_words(va["article"], ratio))
        history = engine.train(spec, tok, model, tr_data, va_data, d / "ckpt", job.seed, deadline,
                               log=lambda m: log(f"[{job.name}] {m}"))
    t_train = time.time() - t0
    preds = engine.generate(spec, tok, model, te["article"].tolist(), target_words(te["article"], ratio),
                            log=lambda m: log(f"[{job.name}] {m}"))
    t_gen = time.time() - t0 - t_train
    pd.DataFrame({"doc_id": te["doc_id"], "prediction": preds}).to_json(
        d / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    scores = score_predictions(te["article"].tolist(), te["summary"].tolist(), preds)
    scores.insert(0, "doc_id", te["doc_id"].to_numpy())
    scores.insert(1, "category", te["category"].to_numpy())
    scores.to_csv(d / "scores.csv", index=False)

    if save_models and spec.finetune and job.seed == 0 and job.setting == "main":
        engine.save_release(spec, tok, model, out / "models" / job.model,
                            {"model": job.model, "base": spec.hf_id, "seed": job.seed,
                             "newssum_version": __version__})
    result = {
        "job": job.name, **asdict(job), "spec": {k: v for k, v in asdict(spec).items() if k != "extra"},
        "n_train": len(tr), "n_val": len(va), "n_test": len(te), "ratio": ratio, "history": history,
        "minutes_train": round(t_train / 60, 2), "minutes_generate": round(t_gen / 60, 2),
        "mean": scores.drop(columns=["doc_id", "category"]).mean().round(4).to_dict(),
        "versions": _versions(), "smoke": smoke,
    }
    (d / "result.json").write_text(json.dumps(result, indent=2))
    shutil.rmtree(d / "ckpt", ignore_errors=True)
    (d / "error.txt").unlink(missing_ok=True)
    log(f"[{job.name}] done: ROUGE-1 {result['mean']['rouge1']:.2f} ROUGE-2 {result['mean']['rouge2']:.2f} "
        f"ROUGE-Lsum {result['mean']['rougeLsum']:.2f} ({(time.time() - t0) / 60:.1f} min)")
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _versions() -> dict:
    import importlib.metadata as md
    v = {"python": platform.python_version()}
    for p in ("torch", "transformers", "peft", "rouge-score", "bert-score", "newssum"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    try:
        import torch
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return v


# ---------------------------------------------------------------- workers

def worker(out: Path, jobs: list[Job], deadline: float, worker_id: str, smoke: bool,
           save_models: bool, hf_overrides: dict | None) -> None:
    from .engine import DeadlineReached

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {worker_id} {msg}", flush=True)

    for job in jobs:
        if is_done(out, job):
            continue
        est = get_spec(job.model).est_minutes * (0.2 if smoke else 1.0)
        # Unfinished jobs with a checkpoint can always start: they save again at the limit.
        has_ckpt = (run_dir(out, job) / "ckpt" / "state.pt").exists()
        if not has_ckpt and time.time() + 60 * est > deadline:
            continue
        if time.time() > deadline - 120:
            break
        if not claim(out, job, worker_id):
            continue
        try:
            run_job(out, job, deadline, smoke, save_models, hf_overrides, log)
        except DeadlineReached:
            log(f"[{job.name}] stopped at the time limit; it will resume in the next session")
            release(out, job)
            break
        except Exception:
            err = traceback.format_exc()
            (run_dir(out, job) / "error.txt").write_text(err)
            log(f"[{job.name}] FAILED\n{err}")
        release(out, job)
    log("worker finished")


def launch(out: str | Path, jobs: list[Job], hours: float, n_gpus: int | None = None,
           smoke: bool = False, save_models: bool = True, hf_overrides: dict | None = None,
           poll: int = 60) -> pd.DataFrame:
    """Run ``jobs`` on all visible GPUs (one worker process per GPU) and wait.

    ``hours`` is the wall-clock budget from now; workers stop starting new jobs
    when a job would not fit, and running jobs checkpoint at the limit.
    """
    out = Path(out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if (out / "runs").exists():
        clear_stale_claims(out)
    deadline = time.time() + hours * 3600
    if n_gpus is None:
        try:
            import torch
            n_gpus = max(torch.cuda.device_count(), 1)
        except ImportError:
            n_gpus = 1
    cfg = {"jobs": [asdict(j) for j in jobs], "deadline": deadline, "smoke": smoke,
           "save_models": save_models, "hf_overrides": hf_overrides or {}}
    (out / "logs" / "jobs.json").write_text(json.dumps(cfg))
    procs, logs = [], []
    for g in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TOKENIZERS_PARALLELISM="false")
        lf = open(out / "logs" / f"worker{g}.log", "a")
        procs.append(subprocess.Popen([sys.executable, "-m", "newssum.runner", "--out", str(out),
                                       "--worker-id", f"gpu{g}"], env=env, stdout=lf, stderr=subprocess.STDOUT))
        logs.append(lf)
        time.sleep(20)  # let the first worker claim before the second starts
    positions = [0] * n_gpus
    while any(p.poll() is None for p in procs):
        time.sleep(poll)
        for g in range(n_gpus):
            path = out / "logs" / f"worker{g}.log"
            with open(path) as fh:
                fh.seek(positions[g])
                new = fh.read()
                positions[g] = fh.tell()
            lines = [ln for ln in new.splitlines() if "] " in ln or "FAILED" in ln or "Error" in ln]
            for ln in lines[-15:]:
                print(ln)
    for g in range(n_gpus):
        with open(out / "logs" / f"worker{g}.log") as fh:
            fh.seek(positions[g])
            print(fh.read()[-3000:])
    for lf in logs:
        lf.close()
    return status(out, jobs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="newssum worker process (started by launch())")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker-id", default="gpu0")
    a = ap.parse_args(argv)
    out = Path(a.out)
    cfg = json.loads((out / "logs" / "jobs.json").read_text())
    jobs = [Job(**j) for j in cfg["jobs"]]
    worker(out, jobs, cfg["deadline"], a.worker_id, cfg["smoke"], cfg["save_models"], cfg["hf_overrides"])


if __name__ == "__main__":
    main()

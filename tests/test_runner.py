import json

import pandas as pd

from newssum.models import REGISTRY
from newssum.pipeline import prepare_data, run_baselines
from newssum.report import collect, main_table, write_all
from newssum.runner import (Job, claim, clear_stale_claims, job_frames, load_prepared, plan, release,
                            run_dir, status)


def test_plan_full_and_smoke():
    jobs = plan("full", seeds=3)
    names = [j.name for j in jobs]
    assert len(names) == len(set(names))
    assert "bart-large-cnn__main__s0" in names and "bart-large-cnn__main__s1" not in names
    assert "bart-base__main__s2" in names and "bart-base__xcat-sport__s0" in names
    assert all(j.model in REGISTRY for j in jobs)
    assert len(plan("smoke")) == 4


def test_claims(tmp_path):
    j = Job("bart-base")
    assert claim(tmp_path, j, "gpu0") and not claim(tmp_path, j, "gpu1")
    release(tmp_path, j)
    assert claim(tmp_path, j, "gpu1")
    assert clear_stale_claims(tmp_path) == 1
    assert status(tmp_path, [j])["status"].iat[0] == "pending"


def test_prepare_baselines_and_report(bbc_root, tmp_path):
    prepare_data(tmp_path, bbc_root.parent)
    df, meta = load_prepared(tmp_path)
    assert 0 < meta["ratio"] < 1 and meta["removed_duplicates"] == 3
    tr, va, te = job_frames(df, Job("bart-base", "xcat-tech"))
    assert (te["category"] == "tech").all() and not (tr["category"] == "tech").any()
    assert set(te["doc_id"]) <= set(df.loc[df["split"] == "test", "doc_id"])

    run_baselines(tmp_path)
    per_doc = collect(tmp_path)
    assert set(per_doc["system"]) == {"lead-3", "lead-ratio", "textrank-ratio", "oracle"}
    t = main_table(per_doc, n_boot=200)
    assert t.iloc[0]["system"] == "oracle"
    assert (t["rougeLsum_lo"] <= t["rougeLsum"]).all()

    # a fake fine-tuned run with two seeds goes through the same report code
    lead = pd.read_csv(run_dir(tmp_path, Job("lead-3")) / "scores.csv", dtype={"doc_id": str})
    for s in (0, 1):
        d = run_dir(tmp_path, Job("bart-base", "main", s))
        d.mkdir(parents=True)
        lead.assign(rougeLsum=lead["rougeLsum"] + s).to_csv(d / "scores.csv", index=False)
        (d / "result.json").write_text(json.dumps({"job": d.name, "model": "bart-base", "setting": "main",
                                                   "seed": s, "history": [{"epoch": 1, "val_loss": 1.0}]}))
    tables = write_all(tmp_path, n_boot=200)
    row = tables["main"].set_index("system").loc["bart-base"]
    assert row["seeds"] == 2 and row["rougeLsum_sd"] > 0
    assert (tmp_path / "results" / "figures" / "rouge_lsum_by_system.png").exists()


def test_oom_retry_halves_batch(tmp_path):
    from newssum.runner import _is_oom, _smaller_batch
    j = Job("flan-t5-base")
    run_dir(tmp_path, j).mkdir(parents=True)
    first = _smaller_batch(tmp_path, j, {})
    assert first == {"batch_size": 2, "grad_accum": 4, "eval_batch": 4}
    assert json.loads((run_dir(tmp_path, j) / "overrides.json").read_text()) == first
    assert _smaller_batch(tmp_path, j, {"batch_size": 1, "grad_accum": 8, "eval_batch": 1}) is None
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate 52.00 MiB"))


def test_prune_outputs_keeps_released_models_only(tmp_path):
    from newssum.runner import prune_outputs
    for m in ("bart-base", "t5-base", "flan-t5-base"):
        (tmp_path / "models" / m).mkdir(parents=True)
        (tmp_path / "models" / m / "model.safetensors").write_bytes(b"x" * 1000)
    done = run_dir(tmp_path, Job("t5-base"))
    (done / "ckpt").mkdir(parents=True)
    (done / "ckpt" / "state.pt").write_bytes(b"x" * 1000)
    (done / "result.json").write_text("{}")
    running = run_dir(tmp_path, Job("flan-t5-large"))
    (running / "ckpt").mkdir(parents=True)
    (running / "ckpt" / "state.pt").write_bytes(b"x")
    prune_outputs(tmp_path)
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["bart-base"]
    assert not (done / "ckpt").exists() and (running / "ckpt" / "state.pt").exists()

"""End-to-end run with tiny random Hugging Face models (CPU, a few minutes).

Skipped unless NEWSSUM_SMOKE=1, because it needs torch, transformers, peft and
internet access. The CI workflow runs it in a separate job.
"""

import json
import os
import time

import pytest

pytestmark = pytest.mark.smoke

if os.environ.get("NEWSSUM_SMOKE") != "1":
    pytest.skip("set NEWSSUM_SMOKE=1 to run", allow_module_level=True)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from newssum.engine import DeadlineReached  # noqa: E402
from newssum.pipeline import prepare_data  # noqa: E402
from newssum.report import collect  # noqa: E402
from newssum.runner import Job, launch, run_dir, run_job  # noqa: E402

TINY = {
    "bart-base": "hf-internal-testing/tiny-random-BartForConditionalGeneration",
    "flan-t5-base": "google/t5-efficient-tiny",
    "qwen2.5-1.5b-zs": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "qwen2.5-1.5b-lora": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}


def test_resume_after_time_limit(bbc_root, tmp_path):
    prepare_data(tmp_path, bbc_root.parent)
    job = Job("bart-base", "main", 0)
    with pytest.raises(DeadlineReached):
        run_job(tmp_path, job, deadline=time.time() - 1, smoke=True, hf_overrides=TINY)
    assert (run_dir(tmp_path, job) / "ckpt" / "state.pt").exists()
    run_job(tmp_path, job, deadline=time.time() + 3600, smoke=True, hf_overrides=TINY)
    r = json.loads((run_dir(tmp_path, job) / "result.json").read_text())
    assert r["history"] and not (run_dir(tmp_path, job) / "ckpt").exists()
    assert (tmp_path / "models" / "bart-base" / "config.json").exists()


def test_launch_all_kinds(bbc_root, tmp_path):
    prepare_data(tmp_path, bbc_root.parent)
    jobs = [Job(m) for m in TINY]
    st = launch(tmp_path, jobs, hours=0.5, n_gpus=1, smoke=True, hf_overrides=TINY, poll=2)
    errors = {p.parent.name: p.read_text()[-2000:] for p in (tmp_path / "runs").glob("*/error.txt")}
    assert not errors, errors
    assert (st["status"] == "done").all(), st
    per_doc = collect(tmp_path, include_smoke=True)
    assert set(per_doc["system"]) == set(TINY)
    assert (tmp_path / "models" / "qwen2.5-1.5b-lora" / "adapter_config.json").exists()

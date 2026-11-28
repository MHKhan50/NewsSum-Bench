"""Rebuild the quantised ONNX copy of the NewsSum BART-base model used by the browser demo.

The first export left the merged decoder's weights in float32 (they sit inside If-branches,
which dynamic quantisation skips unless EnableSubgraph is set), so the demo downloaded about
626 MB. This script exports again from the published checkpoint, quantises every branch, checks
that the quantised model still writes a summary, and only then replaces the files in onnx/.

Run on Kaggle (CPU, Internet on) with the secret HF_TOKEN set:

    python huggingface/requantize_bart.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish  # noqa: E402

REPO = "MHkhan50/newssum-bart-base-bbc"
FILES = ("encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx")
LIMIT_MB = 250

CHECK = '''
import sys
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
d = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(d)
model = ORTModelForSeq2SeqLM.from_pretrained(d)
text = ("The UK economy grew faster than expected in the final three months of the year, official "
        "figures have shown. Gross domestic product rose by 0.7% between October and December. "
        "Analysts had expected growth of 0.5%. The services sector was the main driver of growth.")
summary = tok.decode(model.generate(**tok(text, return_tensors="pt"), max_new_tokens=80)[0], skip_special_tokens=True)
print("Quantised check:", summary)
if len(summary.split()) < 5:
    sys.exit("The quantised model did not write a summary; nothing was uploaded.")
'''


def sizes(api: HfApi) -> dict[str, float]:
    info = api.model_info(REPO, files_metadata=True)
    return {s.rfilename: s.size / 1e6 for s in info.siblings if s.rfilename.startswith("onnx/")}


def main() -> None:
    token = os.environ.get("HF_TOKEN") or sys.exit("Set HF_TOKEN (a Hugging Face token with write access).")
    api = HfApi(token=token)
    print("Now on Hugging Face:", {k: f"{v:.0f} MB" for k, v in sizes(api).items()}, flush=True)

    src = Path(snapshot_download(REPO, ignore_patterns=["onnx/*", "*.md"], token=token))
    out = publish.export_bart(src)

    new = {f: (out / f).stat().st_size / 1e6 for f in FILES}
    total = sum(new.values())
    print("New files:", {k: f"{v:.0f} MB" for k, v in new.items()}, f"total {total:.0f} MB", flush=True)
    if total > LIMIT_MB:
        sys.exit(f"Still {total:.0f} MB (limit {LIMIT_MB}); nothing was uploaded.")

    chk = publish.WORK / "bart_q_check"
    shutil.rmtree(chk, ignore_errors=True)
    shutil.copytree(publish.WORK / "bart32", chk, ignore=shutil.ignore_patterns("*.safetensors", "*.bin"))
    for f in FILES:
        shutil.copy(out / f, chk / f.replace("_quantized", ""))
    script = publish.WORK / "check_bart_q.py"
    script.write_text(CHECK)
    publish.run([str(publish.VENV / "bin" / "python"), str(script), str(chk)])

    for f in FILES:
        api.upload_file(path_or_fileobj=str(out / f), path_in_repo=f"onnx/{f}", repo_id=REPO,
                        commit_message="Quantise every decoder branch (smaller browser download)")
    print("After upload:", {k: f"{v:.0f} MB" for k, v in sizes(api).items()})
    print("ALL DONE")


if __name__ == "__main__":
    main()

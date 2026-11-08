"""Publish the NewsSum V2 checkpoints and the browser demo to Hugging Face.

Run on Kaggle with the NewsSum V2 notebook output attached as input and the
environment variable HF_TOKEN set (a token with write access):

    python huggingface/publish.py

The ONNX export for the browser model runs in a separate virtual environment
(it reuses the system PyTorch) so the notebook's own transformers and
huggingface_hub versions are left untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
INPUT = Path(os.environ.get("NEWSSUM_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("NEWSSUM_WORK", "/kaggle/working"))
VENV = Path("/tmp/onnxenv")
TARGETS = {
    "pegasus-large": ("newssum-pegasus-large-bbc", "pegasus-large.md"),
    "bart-base": ("newssum-bart-base-bbc", "bart-base.md"),
    "qwen2.5-1.5b-lora": ("newssum-qwen2.5-1.5b-lora-bbc", "qwen-lora.md"),
}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_models() -> dict[str, Path]:
    found = {p.parent.name: p.parent for p in INPUT.rglob("newssum/models/*/newssum_meta.json")}
    missing = set(TARGETS) - set(found)
    if missing:
        sys.exit(f"Models not found: {sorted(missing)}. Add the NewsSum V2 notebook output as input.")
    return found


def card(name: str, repo_id: str, user: str) -> bytes:
    text = (HERE / "cards" / name).read_text()
    return text.replace("{repo}", repo_id).replace("{user}", user).encode()


def publish_folder(api: HfApi, user: str, key: str, folder: Path, extra: Path | None = None) -> str:
    repo_name, card_file = TARGETS[key]
    repo_id = f"{user}/{repo_name}"
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), ignore_patterns=["README.md"],
                      commit_message="Upload NewsSum V2 checkpoint")
    if extra is not None:
        api.upload_folder(repo_id=repo_id, folder_path=str(extra), path_in_repo="onnx",
                          commit_message="Add quantised ONNX for Transformers.js")
    api.upload_file(path_or_fileobj=card(card_file, repo_id, user), path_in_repo="README.md", repo_id=repo_id,
                    commit_message="Add model card")
    print("published https://huggingface.co/" + repo_id, flush=True)
    return repo_id


EXPORT = textwrap.dedent('''
    import importlib.util, shutil, sys
    from pathlib import Path

    # Kaggle ships diffusers built for a newer huggingface_hub; optimum imports it
    # if it is present. Hide it: the export below does not need it.
    _find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *a, **k: (
        None if name.split(".")[0] == "diffusers" else _find_spec(name, *a, **k))
    sys.modules["diffusers"] = None
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    src, fp32, onnx_dir, out = map(Path, sys.argv[1:5])
    for d in (fp32, onnx_dir):
        shutil.rmtree(d, ignore_errors=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(src, torch_dtype=torch.float32)
    model.save_pretrained(fp32)
    AutoTokenizer.from_pretrained(src).save_pretrained(fp32)

    from optimum.exporters.onnx import main_export
    main_export(model_name_or_path=str(fp32), output=onnx_dir, task="text2text-generation-with-past")
    print("exported:", sorted(p.name for p in onnx_dir.glob("*.onnx")))

    from onnxruntime.quantization import QuantType, quantize_dynamic
    out.mkdir(parents=True, exist_ok=True)
    for name in ("encoder_model", "decoder_model_merged"):
        f = onnx_dir / f"{name}.onnx"
        if not f.exists():
            sys.exit(f"{f.name} was not exported")
        quantize_dynamic(str(f), str(out / f"{name}_quantized.onnx"), weight_type=QuantType.QInt8)
    for p in sorted(out.iterdir()):
        print(p.name, f"{p.stat().st_size / 1e6:.0f} MB")

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(fp32)
        ort = ORTModelForSeq2SeqLM.from_pretrained(onnx_dir)
        text = ("The UK economy grew faster than expected in the final three months of the year, official "
                "figures have shown. Gross domestic product rose by 0.7% between October and December. "
                "Analysts had expected growth of 0.5%. The services sector was the main driver of growth.")
        ids = tok(text, return_tensors="pt")
        print("ONNX check:", tok.decode(ort.generate(**ids, max_new_tokens=80)[0], skip_special_tokens=True))
    except Exception as e:  # the check is informative only
        print("ONNX check skipped:", e)
''')


def export_bart(src: Path) -> Path:
    """Export BART-base to ONNX and quantise it, in an isolated environment."""
    py = VENV / "bin" / "python"
    if not py.exists():
        run([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(VENV)])
    run([str(py), "-m", "pip", "install", "-q", "optimum[onnxruntime]<2",
         "transformers>=4.40,<4.56", "huggingface_hub<1.0", "onnx", "onnxruntime"])
    script = WORK / "export_bart.py"
    script.write_text(EXPORT)
    out = WORK / "bart_onnx_q"
    run([str(py), str(script), str(src), str(WORK / "bart32"), str(WORK / "bart_onnx"), str(out)])
    return out


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (a Hugging Face token with write access).")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    models = find_models()
    print("Hugging Face user:", user, "| models:", sorted(models), flush=True)

    links = []
    for key in ("pegasus-large", "qwen2.5-1.5b-lora"):
        links.append(publish_folder(api, user, key, models[key]))

    onnx = None
    try:
        onnx = export_bart(models["bart-base"])
    except subprocess.CalledProcessError as e:
        print("ONNX export failed; publishing BART-base without it:", e, flush=True)
    bart = publish_folder(api, user, "bart-base", models["bart-base"], onnx)
    links.append(bart)

    space = f"{user}/newssum-demo"
    if onnx is not None:
        api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
        html = (HERE / "space" / "index.html").read_text().replace("{repo_bart}", bart)
        api.upload_file(path_or_fileobj=html.encode(), path_in_repo="index.html", repo_id=space, repo_type="space",
                        commit_message="NewsSum V2 demo")
        api.upload_file(path_or_fileobj=(HERE / "space" / "README.md").read_bytes(), path_in_repo="README.md",
                        repo_id=space, repo_type="space", commit_message="Space card")

    print("\n=== Done ===")
    if onnx is not None:
        print("Demo:  https://huggingface.co/spaces/" + space)
    for link in links:
        print("Model: https://huggingface.co/" + link)


if __name__ == "__main__":
    main()

"""Model registry. Hyperparameters are fixed in advance and are not tuned on the test set."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short name used in tables and folders
    hf_id: str                    # Hugging Face model id
    kind: str                     # "seq2seq" or "causal"
    finetune: bool = True
    family: str = ""
    prefix: str = ""              # T5-style task prefix
    max_source: int = 1024        # article tokens
    max_target: int = 320         # summary tokens (BBC references are long)
    lr: float = 5e-5
    epochs: int = 8
    patience: int = 2             # early stopping on validation loss
    batch_size: int = 8
    grad_accum: int = 1
    eval_batch: int = 8
    fp16: bool = True             # autocast fp16 (BART, LLMs); False for T5/PEGASUS
    optimizer: str = "adamw"      # "adamw" or "adafactor"
    grad_ckpt: bool = False
    lora: bool = False
    num_beams: int = 4
    est_minutes: float = 20.0     # rough T4 time per run, for scheduling
    save_fp16: bool = True        # store released weights in fp16
    notes: str = ""
    extra: dict = field(default_factory=dict)


LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"

REGISTRY: dict[str, ModelSpec] = {s.key: s for s in [
    # Zero-shot: summarisers trained on CNN/DailyMail, and an instruction-tuned LLM.
    ModelSpec("bart-large-cnn", "facebook/bart-large-cnn", "seq2seq", finetune=False, family="BART",
              eval_batch=8, est_minutes=8, notes="zero-shot, trained on CNN/DailyMail"),
    ModelSpec("pegasus-cnn", "google/pegasus-cnn_dailymail", "seq2seq", finetune=False, family="PEGASUS",
              fp16=False, eval_batch=8, est_minutes=12, notes="zero-shot, trained on CNN/DailyMail"),
    ModelSpec("qwen2.5-1.5b-zs", LLM_ID, "causal", finetune=False, family="Qwen2.5",
              max_source=1024, max_target=400, num_beams=1, eval_batch=8, est_minutes=15,
              notes="zero-shot instruction-tuned LLM"),
    # Fine-tuned encoder-decoders.
    ModelSpec("bart-base", "facebook/bart-base", "seq2seq", family="BART", lr=5e-5, epochs=8,
              batch_size=8, est_minutes=15),
    ModelSpec("t5-base", "google-t5/t5-base", "seq2seq", family="T5", prefix="summarize: ",
              lr=5e-4, optimizer="adafactor", fp16=False, batch_size=4, grad_accum=2,
              grad_ckpt=True, est_minutes=30, save_fp16=False),
    ModelSpec("flan-t5-base", "google/flan-t5-base", "seq2seq", family="Flan-T5", prefix="summarize: ",
              lr=5e-4, optimizer="adafactor", fp16=False, batch_size=4, grad_accum=2,
              grad_ckpt=True, est_minutes=30, save_fp16=False),
    ModelSpec("bart-large", "facebook/bart-large", "seq2seq", family="BART", lr=3e-5, epochs=6,
              batch_size=4, grad_accum=4, eval_batch=4, grad_ckpt=True, est_minutes=45),
    ModelSpec("pegasus-large", "google/pegasus-large", "seq2seq", family="PEGASUS", lr=2e-4,
              optimizer="adafactor", fp16=False, epochs=6, batch_size=2, grad_accum=8, eval_batch=4,
              grad_ckpt=True, est_minutes=100),
    ModelSpec("flan-t5-large", "google/flan-t5-large", "seq2seq", family="Flan-T5", prefix="summarize: ",
              lr=3e-4, optimizer="adafactor", fp16=False, epochs=6, batch_size=2, grad_accum=8,
              eval_batch=4, grad_ckpt=True, est_minutes=110, save_fp16=False),
    # Parameter-efficient fine-tuning of the same LLM.
    ModelSpec("qwen2.5-1.5b-lora", LLM_ID, "causal", family="Qwen2.5", lora=True, lr=2e-4, epochs=3,
              patience=1, max_source=1024, max_target=400, batch_size=2, grad_accum=8, eval_batch=8,
              grad_ckpt=True, num_beams=1, est_minutes=90),
]}

# Cheap jobs first so that a short session still produces useful results.
MAIN_ORDER = ["bart-large-cnn", "pegasus-cnn", "bart-base", "flan-t5-base", "t5-base",
              "qwen2.5-1.5b-zs", "bart-large", "qwen2.5-1.5b-lora", "pegasus-large", "flan-t5-large"]
XCAT_MODEL = "bart-base"


def get_spec(key: str, **overrides) -> ModelSpec:
    spec = REGISTRY[key]
    return replace(spec, **overrides) if overrides else spec

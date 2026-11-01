"""PyTorch fine-tuning and generation.

A small explicit training loop is used instead of ``Seq2SeqTrainer`` so that the
code behaves the same across ``transformers`` versions and so that a run can be
stopped at a time limit and resumed exactly in a later Kaggle session.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .models import ModelSpec

PROMPT = ("Summarise the following news article in about {n} words. Write complete sentences "
          "and keep the key facts, names and figures. Reply with the summary only.\n\n"
          "Article:\n{article}")


class DeadlineReached(Exception):
    """Raised after a checkpoint has been written because the session time is up."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _from_pretrained(cls, name: str, dtype=None):
    if dtype is None:
        return cls.from_pretrained(name)
    try:
        return cls.from_pretrained(name, dtype=dtype)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        return cls.from_pretrained(name, torch_dtype=dtype)


def load(spec: ModelSpec, train: bool):
    """Load tokenizer and model (with LoRA adapters if the spec asks for them)."""
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    dev = device()
    if spec.kind == "causal":
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        model = _from_pretrained(AutoModelForCausalLM, spec.hf_id, dtype)
    else:
        model = _from_pretrained(AutoModelForSeq2SeqLM, spec.hf_id)
    if len(tok) > model.get_input_embeddings().num_embeddings:  # only tiny test models
        model.resize_token_embeddings(len(tok))
    if train and spec.grad_ckpt:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if spec.lora:
        from peft import LoraConfig, get_peft_model
        _skip_old_torchao()
        cfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules="all-linear")
        model = get_peft_model(model, cfg)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    return tok, model.to(dev)


def fit_lengths(spec: ModelSpec, model) -> ModelSpec:
    """Shrink max_source/max_target to what the model's position embeddings allow.

    BART and PEGASUS have learned positions (1024), so longer inputs would index
    past the embedding table. T5 uses relative positions and has no such limit.
    """
    from dataclasses import replace

    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not limit:
        return spec
    if spec.kind == "seq2seq":
        return replace(spec, max_source=min(spec.max_source, limit), max_target=min(spec.max_target, limit))
    room = max(16, limit - spec.max_target - 128)  # prompt template + summary must fit
    return replace(spec, max_source=min(spec.max_source, room))


def _skip_old_torchao() -> None:
    """Some images ship a torchao version that recent PEFT rejects with an
    ImportError even though LoRA on plain nn.Linear layers never uses it."""
    try:
        from peft.import_utils import is_torchao_available
        is_torchao_available()
    except ImportError:
        import peft.tuners.lora.torchao as tao
        tao.is_torchao_available = lambda: False
    except Exception:
        pass


# ---------------------------------------------------------------- prompts and encoding

def build_prompt(tok, article: str, n_target_words: int, max_source: int) -> str:
    ids = tok(article, add_special_tokens=False)["input_ids"][:max_source]
    msg = PROMPT.format(n=n_target_words, article=tok.decode(ids, skip_special_tokens=True))
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True)
    return msg + "\n\nSummary:\n"


def encode(spec: ModelSpec, tok, articles: list[str], summaries: list[str], targets: list[int]) -> list[dict]:
    """Tokenise training pairs. Returns a list of dicts with input_ids and labels."""
    out = []
    if spec.kind == "seq2seq":
        src = tok([spec.prefix + a for a in articles], max_length=spec.max_source, truncation=True)
        lab = tok(text_target=summaries, max_length=spec.max_target, truncation=True)
        for ids, labels in zip(src["input_ids"], lab["input_ids"]):
            out.append({"input_ids": ids, "labels": labels})
        return out
    eos = tok.eos_token or ""
    for a, s, n in zip(articles, summaries, targets):
        p = tok(build_prompt(tok, a, n, spec.max_source), add_special_tokens=False)["input_ids"]
        t = tok(s + eos, add_special_tokens=False)["input_ids"][:spec.max_target]
        out.append({"input_ids": p + t, "labels": [-100] * len(p) + t})
    return out


def collate(items: list[dict], pad_id: int, dev: torch.device) -> dict:
    """Right-pad a batch; labels are padded with -100 (ignored by the loss)."""
    n = max(len(x["input_ids"]) for x in items)
    m = max(len(x["labels"]) for x in items)
    ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    labels = torch.full((len(items), m), -100, dtype=torch.long)
    for i, x in enumerate(items):
        ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
        mask[i, :len(x["input_ids"])] = 1
        labels[i, :len(x["labels"])] = torch.tensor(x["labels"])
    return {"input_ids": ids.to(dev), "attention_mask": mask.to(dev), "labels": labels.to(dev)}


# ---------------------------------------------------------------- training

def _optimizer(spec: ModelSpec, params):
    if spec.optimizer == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=spec.lr, scale_parameter=False, relative_step=False,
                             warmup_init=False, weight_decay=0.0)
        except ImportError:
            return torch.optim.Adafactor(params, lr=spec.lr)
    return torch.optim.AdamW(params, lr=spec.lr, weight_decay=0.01)


def _trainable_state(model, lora: bool) -> dict:
    if lora:
        from peft import get_peft_model_state_dict
        sd = get_peft_model_state_dict(model)
    else:
        sd = model.state_dict()
    return {k: v.detach().to("cpu").clone() for k, v in sd.items()}


def _load_trainable(model, state: dict, lora: bool) -> None:
    if lora:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state)
    else:
        model.load_state_dict(state)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validation_loss(model, data: list[dict], pad_id: int, batch_size: int, use_fp16: bool) -> float:
    model.eval()
    dev = device()
    total, count = 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = collate(data[i:i + batch_size], pad_id, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
            loss = model(**batch).loss
        n = int((batch["labels"] != -100).sum())
        total += float(loss) * n
        count += n
    return total / max(count, 1)


def train(spec: ModelSpec, tok, model, train_data: list[dict], val_data: list[dict], ckpt_dir: Path,
          seed: int, deadline: float, log=print) -> list[dict]:
    """Fine-tune with early stopping on validation loss. Resumable from ``ckpt_dir``.

    Returns the per-epoch history. The best weights are loaded into ``model``.
    Raises ``DeadlineReached`` (after saving state) if ``deadline`` passes.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dev = device()
    use_fp16 = spec.fp16 and dev.type == "cuda"
    pad_id = tok.pad_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _optimizer(spec, params)
    n_batches = math.ceil(len(train_data) / spec.batch_size)
    updates_per_epoch = math.ceil(n_batches / spec.grad_accum)
    total = max(1, updates_per_epoch * spec.epochs)
    warmup = max(1, int(0.05 * total))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    st = {"epoch": 0, "batch": 0, "best_val": float("inf"), "best_epoch": -1, "bad": 0,
          "history": [], "done": False}
    state_path, best_path = ckpt_dir / "state.pt", ckpt_dir / "best.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        _load_trainable(model, saved["model"], spec.lora)
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        st = saved["progress"]
        log(f"resumed at epoch {st['epoch']} batch {st['batch']}")

    def save_state() -> None:
        _atomic_save({"model": _trainable_state(model, spec.lora), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                      "progress": st}, state_path)

    while not st["done"] and st["epoch"] < spec.epochs:
        epoch = st["epoch"]
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(train_data), generator=g).tolist()
        batches = [order[i:i + spec.batch_size] for i in range(0, len(order), spec.batch_size)]
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for b in range(st["batch"], len(batches)):
            batch = collate([train_data[i] for i in batches[b]], pad_id, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss
            scaler.scale(loss / spec.grad_accum).backward()
            running += float(loss)
            seen += 1
            if (b + 1) % spec.grad_accum == 0 or b == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if time.time() > deadline and b < len(batches) - 1:
                    st["batch"] = b + 1
                    save_state()
                    log(f"time limit: saved checkpoint at epoch {epoch} batch {b + 1}")
                    raise DeadlineReached
        val = validation_loss(model, val_data, pad_id, spec.eval_batch, use_fp16)
        rec = {"epoch": epoch + 1, "train_loss": running / max(seen, 1), "val_loss": val,
               "minutes": (time.time() - t0) / 60}
        st["history"].append(rec)
        if val < st["best_val"] - 1e-4:
            st.update(best_val=val, best_epoch=epoch + 1, bad=0)
            _atomic_save(_trainable_state(model, spec.lora), best_path)
        else:
            st["bad"] += 1
        log(f"epoch {epoch + 1}/{spec.epochs} train {rec['train_loss']:.4f} val {val:.4f}"
            f"{' *' if st['best_epoch'] == epoch + 1 else ''}")
        if not math.isfinite(val):
            raise FloatingPointError(f"validation loss is {val}; training diverged")
        st.update(epoch=epoch + 1, batch=0)
        if st["bad"] >= spec.patience:
            st["done"] = True
            log(f"early stopping, best epoch {st['best_epoch']}")
        save_state()
        if time.time() > deadline and not st["done"] and st["epoch"] < spec.epochs:
            log("time limit: saved checkpoint at end of epoch")
            raise DeadlineReached

    _load_trainable(model, torch.load(best_path, map_location="cpu", weights_only=False), spec.lora)
    return st["history"]


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate(spec: ModelSpec, tok, model, articles: list[str], targets: list[int], log=print) -> list[str]:
    """Summaries in input order. Beam search for seq2seq, greedy decoding for LLMs."""
    model.eval()
    dev = device()
    use_fp16 = spec.fp16 and dev.type == "cuda"
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.use_cache = True
    order = sorted(range(len(articles)), key=lambda i: -len(articles[i]))
    out: list[str] = [""] * len(articles)
    bs = spec.eval_batch
    for start in range(0, len(order), bs):
        idx = order[start:start + bs]
        if spec.kind == "seq2seq":
            enc = tok([spec.prefix + articles[i] for i in idx], max_length=spec.max_source,
                      truncation=True, padding=True, return_tensors="pt").to(dev)
            kw = dict(num_beams=spec.num_beams, max_new_tokens=spec.max_target, no_repeat_ngram_size=3)
            if spec.num_beams > 1:
                kw["early_stopping"] = True
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
                gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **kw)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
        else:
            prompts = [build_prompt(tok, articles[i], targets[i], spec.max_source) for i in idx]
            enc = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
                gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                     max_new_tokens=spec.max_target, do_sample=False, num_beams=1,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = " ".join(t.replace("<n>", " ").split())  # PEGASUS-CNN marks newlines as <n>
        if (start // bs) % 10 == 0:
            log(f"generated {min(start + bs, len(order))}/{len(order)}")
    return out


def save_release(spec: ModelSpec, tok, model, path: Path, meta: dict) -> None:
    """Save weights for publishing (LoRA: adapter only)."""
    path.mkdir(parents=True, exist_ok=True)
    if spec.lora:
        model.save_pretrained(path)
    else:
        if spec.save_fp16:
            model = model.half()
        model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    (path / "newssum_meta.json").write_text(json.dumps(meta, indent=2))

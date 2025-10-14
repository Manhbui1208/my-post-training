import os
import math
import dataclasses
import json
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset
import copy
import types
import torch.nn as nn


import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    GenerationConfig,
)
from transformers.generation.logits_process import LogitsProcessorList, TopPLogitsWarper, InfNanRemoveLogitsProcessor
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32  
torch.manual_seed(0)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_JSONL = "data/tldr_toy_240/train.jsonl"  
VALID_JSONL = "data/tldr_toy_240/valid.jsonl"
BATCH_SIZE = 2          
MAX_INPUT_TOKENS = 384  
MAX_NEW_TOKENS = 24     

def to_device(model, device=DEVICE):
    return model.to(device)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (
                r.get("prompt")
                or r.get("post")
                or (r.get("messages", [{}])[0].get("content") if r.get("messages") else None)
                or ""
            )
            text = (text or "").strip()
            if not text:
                continue
            rows.append({"input": text, "gold": r.get("summary", "")})
    return rows

def batch_encode_from_ds(ds, batch_size=BATCH_SIZE):
    idx = torch.randint(0, len(ds), (batch_size,))
    chats = []
    for i in idx.tolist():
        sample = ds[i]              # {'input': ..., 'gold': ...}
        msgs = [
            {"role": "system", "content": "You are a helpful assistant. Write a concise TL;DR in 1–2 sentences."},
            {"role": "user", "content": sample["input"]},
        ]
        chats.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_INPUT_TOKENS).to(DEVICE)
    return enc, idx

def ppo_collate(samples):
    # samples: list[{"input": str, "gold": str}]
    chats, golds = [], []
    for s in samples:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant. Write a concise TL;DR in 1–2 sentences."},
            {"role": "user",   "content": s["input"]},
        ]
        chats.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        golds.append(s.get("gold", ""))

    enc = tok(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "golds": golds,  
    }
    return batch

class PromptDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                text = (r.get("prompt") or r.get("post") or
                        (r.get("messages", [{}])[0].get("content") if r.get("messages") else "") or "").strip()
                if text:
                    self.rows.append({"input": text, "gold": r.get("summary", "")})

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]  # return {input, gold}

class HiddenStateRewardModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.scorer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.scorer.bias)

    @torch.no_grad()
    def score(self, last_hidden_state: torch.Tensor):
        token_scores = self.scorer(last_hidden_state).squeeze(-1)  # (B, T, 1) -> (B, T)
        return token_scores
    
class ScoringWrapper(nn.Module):
    """
    - forward(...) gọi vào policy (đã freeze) với output_hidden_states=True
    - score(last_hidden_state) dùng HiddenStateRewardModel để chấm
    """
    def __init__(self, base_model: nn.Module, scorer_model: HiddenStateRewardModel):
        super().__init__()
        self.base_model_prefix = "base_model"
        self.base_model = base_model
        self.scorer_model = scorer_model
        # Freeze base_model để TRL không “nghĩ” nó train phần này
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        # một số bản TRL đọc .config từ reward_model
        self.config = getattr(base_model, "config", None)

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        # đảm bảo device/dtype nhất quán
        return self.base_model(*args, **kwargs)

    @torch.no_grad()
    def score(self, last_hidden_state: torch.Tensor):
        return self.scorer_model.score(last_hidden_state)
    
class SafePPOTrainer(PPOTrainer):
    def __init__(self, *args, generation_config=None, logits_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._safe_gen_cfg = generation_config       
        self._safe_logits_processor = logits_processor 

    def _safe_generate_impl(self, batch):
        gen_cfg = self._safe_gen_cfg or getattr(self.model, "generation_config", None)
        return self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=gen_cfg,
            logits_processor=self._safe_logits_processor,
        )

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    tok.truncation_side = "left"

    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=DTYPE)
    value_model  = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=DTYPE)
    ref_model    = copy.deepcopy(policy_model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    for m in (policy_model, value_model, ref_model):
        m.to(DEVICE)
        m.config.pad_token_id = tok.pad_token_id
        m.config.eos_token_id = tok.eos_token_id
        m.config.use_cache = True
    if not hasattr(value_model, "base_model_prefix"):
        value_model.base_model_prefix = "pretrained_model"
    if not hasattr(value_model, "score"):
        def _value_score(self, last_hidden_state):
            # v_head: Linear(H→1) áp cho (B,T,H) → (B,T,1)
            return self.v_head(last_hidden_state).squeeze(-1)
        value_model.score = types.MethodType(_value_score, value_model)

    hidden_size = policy_model.config.hidden_size
    scorer = HiddenStateRewardModel(hidden_size).to(DEVICE)
    reward_proxy = ScoringWrapper(policy_model, scorer).to(DEVICE)

    logits_processors = LogitsProcessorList([
        InfNanRemoveLogitsProcessor(),
        TopPLogitsWarper(0.98, min_tokens_to_keep=1),
    ])

    gen_cfg = GenerationConfig(
        do_sample=True, temperature=1.0,
        top_p=0.98, top_k=0,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        renormalize_logits=True,
    )

    def _safe_generate(self, *args, **kwargs):
        if "logits_processor" not in kwargs or kwargs["logits_processor"] is None:
            kwargs["logits_processor"] = logits_processors
        gen_cfg = kwargs.get("generation_config", getattr(self, "generation_config", None))
        if gen_cfg is not None:
            gen_cfg.renormalize_logits = True
            if gen_cfg.top_p is None: gen_cfg.top_p = 0.98
            if gen_cfg.do_sample is None: gen_cfg.do_sample = True
            if gen_cfg.top_k is None: gen_cfg.top_k = 0
        else:
            kwargs["generation_config"] = self.generation_config
            kwargs["generation_config"].renormalize_logits = True
        return _orig_generate(*args, **kwargs)
    _orig_generate = policy_model.generate
    policy_model.generate = types.MethodType(_safe_generate, policy_model)
    policy_model.generation_config = gen_cfg
    train_ds = PromptDataset(TRAIN_JSONL)

    ppo_cfg = PPOConfig(
        # output_dir="runs/ppo_toy",
        # report_to=["tensorboard"],
        run_name="mps_ppo_toy",
        logging_strategy="steps",
        log_level="info",
        logging_steps=1,
        learning_rate=2e-6,
        batch_size=BATCH_SIZE,
        mini_batch_size=1,
        temperature=1.0,
        num_ppo_epochs=1,
        kl_coef=0.05,
        cliprange=0.2,
        cliprange_value=0.2,
        num_sample_generations=0,
        vf_coef=0.2,
        gamma=1.0,
        lam=0.95,
        eval_strategy="no",
        include_tokens_per_second=True,
    )
    ppo = SafePPOTrainer(
        args=ppo_cfg,
        processing_class=tok,      # tokenizer
        model=policy_model,               # policy
        ref_model=ref_model,       # freeze
        reward_model=reward_proxy, 
        train_dataset=train_ds,    
        value_model=value_model,
        generation_config=gen_cfg,
        data_collator=ppo_collate,         
        logits_processor=logits_processors
    )
    ppo.train()
    



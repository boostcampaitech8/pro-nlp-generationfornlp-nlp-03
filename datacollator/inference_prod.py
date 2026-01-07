import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional

from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

from config import get_config
from data_utils import load_data, process_dataset_for_inference


# -----------------------------
# 0) checkpoint에서 base_model 자동 추출
# -----------------------------
def get_base_model_from_checkpoint(checkpoint_dir: str) -> Optional[str]:
    cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path", None)
    except Exception:
        return None


# -----------------------------
# 1) tokenizer template 보고 system merge 필요 여부
# -----------------------------
def need_system_merge(tokenizer) -> bool:
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        return False
    return "System role not supported" in tpl


def normalize_messages(messages: List[Dict], merge_system: bool) -> List[Dict]:
    """
    merge_system=True이면 system role 제거하고 첫 user에 system 내용을 prepend.
    """
    if not merge_system or not messages:
        return messages

    msgs = list(messages)
    if msgs[0].get("role") != "system":
        return msgs

    system_text = (msgs[0].get("content") or "").strip()
    rest = msgs[1:]

    if not system_text:
        return rest

    if len(rest) == 0:
        return [{"role": "user", "content": system_text}]

    if rest[0].get("role") == "user":
        rest = rest.copy()
        rest[0] = {**rest[0], "content": system_text + "\n\n" + (rest[0].get("content") or "")}
        return rest

    # system 다음이 user가 아닌 경우 방어
    return [{"role": "user", "content": system_text}] + rest


def build_chat_text(tokenizer, messages: List[Dict]) -> str:
    merge_system = need_system_merge(tokenizer)
    msgs = normalize_messages(messages, merge_system=merge_system)
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )


# -----------------------------
# 2) 정답 토큰(1~5) 준비 (fallback 포함)
# -----------------------------
def get_answer_token_ids(tokenizer):
    """
    어떤 tokenizer는 "1"이 한 토큰이 아닐 수 있음.
    -> ["1","2","3","4","5"] 실패하면 [" 1",...], ["\n1",...]로 fallback.
    """
    candidates = [
        ["1", "2", "3", "4", "5"],
        [" 1", " 2", " 3", " 4", " 5"],
        ["\n1", "\n2", "\n3", "\n4", "\n5"],
    ]
    for cand in candidates:
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in cand]
        if all(len(x) == 1 for x in ids):
            return [x[0] for x in ids]
    raise ValueError("정답(1~5) 토큰이 단일 토큰으로 매핑되지 않습니다. generate 방식으로 바꿔야 해요.")


# -----------------------------
# 3) Logits 기반 추론
# -----------------------------
@torch.no_grad()
def inference_logits_and_probs(model, input_ids, attention_mask, choice_token_ids):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    last_pos = attention_mask.long().sum(dim=1) - 1
    last_pos = torch.clamp(last_pos, min=0)

    pred_idx = []
    probs = []
    for i in range(input_ids.size(0)):
        pos = last_pos[i].item()
        ans_logits = logits[i, pos, choice_token_ids]          # (5,)
        ans_probs = torch.softmax(ans_logits.float(), dim=-1)  # (5,)
        pred = torch.argmax(ans_probs).item()                  # 0~4
        pred_idx.append(pred)
        probs.append(ans_probs.detach().cpu().tolist())
    return pred_idx, probs


# -----------------------------
# 4) 모델/토크나이저 로드
#    - base_model 우선순위:
#        (1) checkpoint의 adapter_config.json base_model_name_or_path
#        (2) config.model.model_name
# -----------------------------
def load_model_and_tokenizer(config, checkpoint_dir: str):
    base_from_ckpt = get_base_model_from_checkpoint(checkpoint_dir)
    base_model = base_from_ckpt or config.model.model_name

    print("\n🤖 모델 로드")
    print(f"  - base_model: {base_model}")
    print(f"  - checkpoint: {checkpoint_dir}")

    # 1) base model 로드 (Unsloth)
    model, _tok = FastLanguageModel.from_pretrained(
        base_model,
        dtype=torch.float16,              # V100 안정
        load_in_4bit=True,                # QLoRA 추론 메모리 절약
        max_seq_length=config.training.max_seq_length,
    )

    # 2) LoRA adapter 로드
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()

    # 3) tokenizer는 checkpoint에 저장된 거 우선 (없으면 base_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        print("  - tokenizer: from checkpoint")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        print("  - tokenizer: from base model")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  ✅ 로드 완료")
    return model, tokenizer, base_model


# -----------------------------
# 5) VALID / TEST
# -----------------------------
def run_valid(config, checkpoint_dir: str, save_valid_pred_csv: Optional[str] = None):
    print("=" * 60)
    print("📊 VALID inference (macro-F1)")
    print("=" * 60)

    model, tokenizer, base_model = load_model_and_tokenizer(config, checkpoint_dir)
    choice_token_ids = get_answer_token_ids(tokenizer)

    df_valid = load_data(config.path.valid_data)
    valid_data = process_dataset_for_inference(df_valid)
    valid_data = [d for d in valid_data if d.get("label") is not None]

    device = model.device
    bs = config.training.per_device_eval_batch_size

    all_ids, all_preds, all_labels = [], [], []
    all_probs = []

    for i in tqdm(range(0, len(valid_data), bs), desc="Valid"):
        batch = valid_data[i:i+bs]

        batch_texts, batch_ids, batch_labels = [], [], []
        for item in batch:
            batch_texts.append(build_chat_text(tokenizer, item["messages"]))
            batch_ids.append(item["id"])
            batch_labels.append(int(item["label"]) - 1)  # 1~5 -> 0~4

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.training.max_seq_length,
        ).to(device)

        pred_idx, probs = inference_logits_and_probs(
            model, enc["input_ids"], enc["attention_mask"], choice_token_ids
        )

        all_ids.extend(batch_ids)
        all_preds.extend(pred_idx)
        all_labels.extend(batch_labels)
        all_probs.extend(probs)

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)

    print(f"\n✅ base_model: {base_model}")
    print(f"- Macro-F1: {macro_f1:.6f}\n- Accuracy: {acc:.6f}")

    if save_valid_pred_csv:
        os.makedirs(os.path.dirname(save_valid_pred_csv), exist_ok=True)
        out = pd.DataFrame({
            "id": all_ids,
            "pred": [p + 1 for p in all_preds],
            "p1": [p[0] for p in all_probs],
            "p2": [p[1] for p in all_probs],
            "p3": [p[2] for p in all_probs],
            "p4": [p[3] for p in all_probs],
            "p5": [p[4] for p in all_probs],
            "label": [y + 1 for y in all_labels],
            "checkpoint": os.path.basename(checkpoint_dir.rstrip("/")),
            "base_model": base_model,
        })
        out.to_csv(save_valid_pred_csv, index=False)
        print(f"🧾 valid probs 저장: {save_valid_pred_csv}")


def run_test(config, checkpoint_dir: str, output_csv: str, save_probs_csv: Optional[str] = None):
    print("=" * 60)
    print("🧾 TEST inference (submission 생성)")
    print("=" * 60)

    model, tokenizer, base_model = load_model_and_tokenizer(config, checkpoint_dir)
    choice_token_ids = get_answer_token_ids(tokenizer)

    df_test = load_data(config.path.test_data)
    test_data = process_dataset_for_inference(df_test)

    device = model.device
    bs = config.training.per_device_eval_batch_size

    ids, answers = [], []
    probs_rows = []

    for i in tqdm(range(0, len(test_data), bs), desc="Test"):
        batch = test_data[i:i+bs]

        batch_texts, batch_ids = [], []
        for item in batch:
            batch_texts.append(build_chat_text(tokenizer, item["messages"]))
            batch_ids.append(item["id"])

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.training.max_seq_length,
        ).to(device)

        pred_idx, probs = inference_logits_and_probs(
            model, enc["input_ids"], enc["attention_mask"], choice_token_ids
        )

        ids.extend(batch_ids)
        answers.extend([p + 1 for p in pred_idx])

        if save_probs_csv:
            for _id, pr, pb in zip(batch_ids, pred_idx, probs):
                probs_rows.append([_id, pr + 1, pb[0], pb[1], pb[2], pb[3], pb[4], os.path.basename(checkpoint_dir.rstrip("/")), base_model])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sub = pd.DataFrame({"id": ids, "answer": answers})
    sub.to_csv(output_csv, index=False)
    print(f"\n✅ submission 저장: {output_csv}")

    if save_probs_csv:
        os.makedirs(os.path.dirname(save_probs_csv), exist_ok=True)
        dfp = pd.DataFrame(probs_rows, columns=["id", "pred", "p1", "p2", "p3", "p4", "p5", "checkpoint", "base_model"])
        dfp.to_csv(save_probs_csv, index=False)
        print(f"✅ soft voting용 probs 저장: {save_probs_csv}")


def main():
    config = get_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], required=True)
    ap.add_argument("--checkpoint_dir", type=str, required=True)

    ap.add_argument("--output_csv", type=str, default=config.path.output_csv)
    ap.add_argument("--save_probs_csv", type=str, default=None)
    ap.add_argument("--save_valid_pred_csv", type=str, default=None)
    args = ap.parse_args()

    if args.mode == "valid":
        run_valid(config, args.checkpoint_dir, save_valid_pred_csv=args.save_valid_pred_csv)
    else:
        run_test(config, args.checkpoint_dir, args.output_csv, save_probs_csv=args.save_probs_csv)


if __name__ == "__main__":
    main()

"""
학습 스크립트
Usage:
    python train.py
    python train.py --exp large_context
"""
import sys
import unsloth
import os
import argparse
import torch
import numpy as np
import random
import json
import evaluate

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

from torch.utils.data import DataLoader

# from code.config import get_config, get_experiment_config
from config import get_config, get_experiment_config
from data_utils import (
    load_data,
    process_dataset_for_training,
    setup_tokenizer,
    tokenize_dataset,
    get_token_statistics
)


# =============================================================================
# 유틸리티 함수
# =============================================================================
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback

class AddStepToLogsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # logs dict에 step 추가
        logs["step"] = state.global_step
        # 보기 좋게 한 줄 출력(원하면 삭제)
        print(logs)

def _find_sublist(lst, sub):
    """lst 안에서 sub가 처음 등장하는 시작 인덱스 반환, 없으면 -1"""
    n, m = len(lst), len(sub)
    if m == 0 or n < m:
        return -1
    for i in range(n - m + 1):
        if lst[i:i+m] == sub:
            return i
    return -1

# ============================ CODE REVIEW REQUEST =============================
# Review focus:
# 1) CompletionOnlyDataCollator가 정답 번호만 잘 마스킹해서 loss 계산에 들어가는지 
# 2) (설명이 가능하시다면) SFTTrainer에 DataCollator을 안넣어도 성능 차이가 없는데 어떤 차이인지.. 
# ==============================================================================
class CompletionOnlyDataCollator:
    """
    response_template(기본: <|im_start|>assistant) 이후의 답변에서
    '1~5' 정답 토큰 1개만 labels로 남기고 나머지는 -100 마스킹.
    => loss/metric 둘 다 안정화됨 (객관식 분류에 최적)
    """
    def __init__(self, tokenizer, response_template="<|im_start|>assistant", ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.pad_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

        # response template ids
        self.response_ids = tokenizer.encode(response_template, add_special_tokens=False)
        if len(self.response_ids) == 0:
            raise ValueError("response_template이 토큰화되지 않았어요. 템플릿 문자열을 확인해줘!")

        # <|im_end|> 토큰 id (있으면 마스킹용)
        end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_id = end_ids[0] if len(end_ids) == 1 else None

        # ✅ 1~5 토큰 id (반드시 1토큰이어야 logits 방식이 깔끔함)
        self.choice_token_ids = []
        for s in ["1", "2", "3", "4", "5"]:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"'{s}'가 1 토큰이 아닙니다: {ids} (tokenizer 변경/공백/템플릿 확인 필요)")
            self.choice_token_ids.append(ids[0])

        print(f"✓ Response template: {response_template}")
        print(f"✓ Response template IDs: {self.response_ids}")
        print(f"✓ im_end_id: {self.im_end_id}")
        print(f"✓ choice_token_ids(1~5): {self.choice_token_ids}")

    def __call__(self, features):
        batch = self.pad_collator(features)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index  # pad는 loss 제외

        for i in range(input_ids.size(0)):
            ids_list = input_ids[i].tolist()

            start = _find_sublist(ids_list, self.response_ids)
            if start == -1:
                labels[i, :] = self.ignore_index
                continue

            end = start + len(self.response_ids)  # assistant 시작 직후

            # 1) 프롬프트 전체 마스킹
            labels[i, :end] = self.ignore_index

            # 2) assistant 이후에서 "1~5"가 처음 등장하는 위치를 찾는다
            ans_idx = None
            for j in range(end, len(ids_list)):
                if ids_list[j] in self.choice_token_ids:
                    ans_idx = j
                    break

            if ans_idx is None:
                # 정답 숫자 토큰이 없으면 학습 신호 제거
                labels[i, :] = self.ignore_index
                continue

            # 3) ✅ 정답 1토큰만 남기고 나머지는 전부 마스킹
            labels[i, :ans_idx] = self.ignore_index
            labels[i, ans_idx+1:] = self.ignore_index

            # 4) (선택) im_end는 마스킹
            if self.im_end_id is not None:
                labels[i, labels[i] == self.im_end_id] = self.ignore_index

        batch["labels"] = labels
        return batch

def set_seed(seed: int):
    """재현성을 위한 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print(f"✓ Seed 고정: {seed}")


def get_torch_dtype(dtype_str: str):
    """문자열을 torch dtype으로 변환"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float16)

# ================================
# 설정/학습 파라미터 출력용
# ================================
def pretty_print_config(config):
    """config.py에서 불러온 원본 설정값(의도한 값) 출력"""
    print("\n" + "=" * 80)
    print("📋 CONFIG (from config.py)")
    print("=" * 80)
    try:
        print(json.dumps(asdict(config), indent=2, ensure_ascii=False))
    except Exception as e:
        print("❌ config 출력 실패:", e)
        print(config)
    print("=" * 80 + "\n")


def print_trainer_args(trainer):
    """trainer.args에 실제로 적용된 값(진짜 적용값) 출력"""
    print("\n" + "=" * 80)
    print("⚙️ TRAINER ARGS (actual applied)")
    print("=" * 80)

    keys = [
        # 배치/스텝
        "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "num_train_epochs", "max_steps",
        # lr / scheduler
        "learning_rate", "weight_decay", "lr_scheduler_type", "warmup_ratio", "warmup_steps",
        # precision
        "fp16", "bf16",
        # logging / save / eval
        "logging_steps", "save_strategy", "save_steps", "save_total_limit",
        "evaluation_strategy", "eval_strategy", "eval_steps",
        # best model
        "load_best_model_at_end", "metric_for_best_model", "greater_is_better",
        # output
        "output_dir",
    ]

    for k in keys:
        if hasattr(trainer.args, k):
            print(f"{k:>24}: {getattr(trainer.args, k)}")

    print("=" * 80 + "\n")


# =============================================================================
# 메트릭 함수
# =============================================================================
def create_metric_functions(tokenizer):
    # ✅ 1~5 토큰 id 만들기 (반드시 1토큰이어야 함)
    choice_token_ids = []
    for s in ["1", "2", "3", "4", "5"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"'{s}' is not 1 token: {ids}")
        choice_token_ids.append(ids[0])

    f1_metric = evaluate.load("f1")

    def preprocess_logits_for_metrics(logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]  # (B, T, V)
        B, T, V = logits.shape

        labels_t = labels
        ans_pos = []
        for i in range(B):
            positions = (labels_t[i] != -100).nonzero(as_tuple=False).squeeze(-1)
            ans_pos.append(int(positions[0].item()) if positions.numel() > 0 else T - 1)
        ans_pos = torch.tensor(ans_pos, device=logits.device)

        batch_idx = torch.arange(B, device=logits.device)
        ans_logits = logits[batch_idx, ans_pos, :]              # (B, V)

        # (B, 5)
        ans_logits_5 = ans_logits[:, choice_token_ids]
        return ans_logits_5

    def compute_metrics(eval_pred):
        logits_5, labels = eval_pred  # logits_5: (B,5), labels: (B,T)

        B, T = labels.shape
        y_true = []
        valid_idx = []

        for i in range(B):
            positions = np.where(labels[i] != -100)[0]
            if len(positions) == 0:
                continue
            tid = int(labels[i, positions[0]])
            if tid in choice_token_ids:
                y_true.append(choice_token_ids.index(tid))  # 0~4
                valid_idx.append(i)

        if len(valid_idx) == 0:
            return {"f1": 0.0}

        logits_5_valid = logits_5[valid_idx]
        y_pred = np.argmax(logits_5_valid, axis=-1)

        # macro f1
        return f1_metric.compute(predictions=y_pred, references=y_true, average="macro")

    # ✅ 여기 return이 "반드시" 함수 최하단에 있어야 함!!
    return preprocess_logits_for_metrics, compute_metrics


# =============================================================================
# 메인 학습 함수
# =============================================================================

def train(config):
    """모델 학습 실행"""

    print("=" * 60)
    print("🚀 학습 시작")
    print("=" * 60)

    # 1. 시드 고정
    set_seed(config.training.seed)
    pretty_print_config(config)

    # 2. 데이터 로드 및 전처리
    print("\n📂 데이터 로드 중...")
    # ✅ train / valid 데이터 둘 다 로드
    df_train = load_data(config.path.train_data)
    df_valid = load_data(config.path.valid_data)

    train_dataset_raw = process_dataset_for_training(df_train)
    valid_dataset_raw = process_dataset_for_training(df_valid)

    print(f"  - Train samples: {len(train_dataset_raw)}")
    print(f"  - Valid samples: {len(valid_dataset_raw)}")

    # print(f"💽 Train Data Format: {train_dataset_raw['messages'][0:2]}")
    # print(f"💽 Valid Data Format: {valid_dataset_raw['messages'][0:2]}")

    # 3. 모델 및 토크나이저 로드
    print(f"\n🤖 모델 로드 중: {config.model.model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        config.model.model_name,
        dtype=get_torch_dtype(config.model.torch_dtype),
        # trust_remote_code=config.model.trust_remote_code,
        load_in_4bit=True,
        max_seq_length=config.training.max_seq_length,   # ✅ 추가
    )

    tokenizer = setup_tokenizer(tokenizer)

    print("  - 모델 및 토크나이저 로드 완료")

    # 4. 토큰화
    print(f"\n📝 토큰화 중 (max_seq_length: {config.training.max_seq_length})...")
    # tokenized_dataset = tokenize_dataset(
    #     processed_dataset,
    #     tokenizer,
    #     max_seq_length=config.training.max_seq_length
    # )

    tokenized_train = tokenize_dataset(
        train_dataset_raw,
        tokenizer,
        max_seq_length=config.training.max_seq_length
    )

    tokenized_valid = tokenize_dataset(
        valid_dataset_raw,
        tokenizer,
        max_seq_length=config.training.max_seq_length
    )


    """

    # Train/Eval 분할
    split_dataset = tokenized_dataset.train_test_split(
        test_size=config.training.test_size,
        seed=config.training.seed
    )
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Eval: {len(eval_dataset)} samples")

    """

    # 토큰 통계
    # stats = get_token_statistics(tokenized_dataset, tokenizer)
    # print(f"  - Token 길이: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}")

    stats = get_token_statistics(tokenized_train, tokenizer)
    print(f"  - Token 길이(train): min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}")

    # print(tokenized_dataset.column_names)
    # print(tokenized_dataset[0].keys())

    # 5. LoRA 설정
    print(f"\n⚙️ LoRA 설정")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,# ✅ 여기 2개 추가
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,  # "unsloth"
        use_rslora=config.lora.use_rslora,

        random_state=config.training.seed,
        max_seq_length=config.training.max_seq_length,
        loftq_config=None
    )
    print(f"  - r: {config.lora.r}, alpha: {config.lora.lora_alpha}")
    print(f"  - target_modules: {config.lora.target_modules}")


    # 6. 메트릭 함수 생성
    preprocess_logits_for_metrics, compute_metrics = create_metric_functions(tokenizer)

    # 7. SFTConfig 설정
    print(f"\n📋 학습 설정")
    print(f"  - epochs: {config.training.num_train_epochs}")
    print(f"  - batch_size: {config.training.per_device_train_batch_size}")
    print(f"  - learning_rate: {config.training.learning_rate}")
    print(f"  - output_dir: {config.path.output_dir}")

    sft_config = SFTConfig(
        # do_train=True,
        # do_eval=True,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_steps=config.training.warmup_steps,
        warmup_ratio=config.training.warmup_ratio,
        max_steps=config.training.max_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.training.logging_steps,
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        seed = config.training.seed,
        output_dir=config.path.output_dir,

        per_device_eval_batch_size=config.training.per_device_eval_batch_size,

        # # ✅ 저장/평가/베스트
        # save_strategy="epoch",
        # eval_strategy="epoch",                 # transformers 최신은 eval_strategy
        # save_total_limit=2,                    # best 날아가는 것 방지
        # metric_for_best_model="eval_f1",
        # greater_is_better=True,
        # load_best_model_at_end=True,

        # ✅ config 값 그대로 사용
        save_strategy=config.training.save_strategy,
        eval_strategy=config.training.eval_strategy,   # 너 코드가 eval_strategy 쓰고 있으니 그대로
        eval_steps = config.training.eval_steps,          # 추천: 100~300 사이
        save_steps = config.training.save_steps,          # eval_steps랑 동일하게

        save_total_limit=config.training.save_total_limit, # best + last 남기려면 2 이상


        # ✅ best 모델 저장 (네가 추가한 3개가 여기로 들어와야 적용됨)
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        save_only_model=config.training.save_only_model,

    )

    data_collator = CompletionOnlyDataCollator(
        tokenizer,
        response_template="<|im_start|>assistant"
    )
    # 8. Trainer 생성
    trainer = SFTTrainer(
        model=model,
        tokenizer = tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=sft_config,
        callbacks=[AddStepToLogsCallback()],
    )
    print("collator:", type(trainer.data_collator))
    print_trainer_args(trainer)

    # DataCollator 검증용 샘플 배치 출력
    dl = DataLoader(
        tokenized_train.select(range(1)),
        batch_size=1,
        collate_fn=trainer.data_collator,
    )
    batch = next(iter(dl))

    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    loss_pos = [i for i, t in enumerate(labels) if t != -100]
    loss_tokens = [input_ids[i] for i in loss_pos]
    loss_text = tokenizer.decode(loss_tokens, skip_special_tokens=False)

    print("\n" + "="*80)
    print("🔎 FULL INPUT:")
    print(full_text)
    print("-"*80)
    print("🔎 LOSS TOKENS ONLY:")
    print(loss_text)
    print("="*80 + "\n")


    # 9. 학습 실행
    print("\n" + "=" * 60)
    print("🏃 학습 실행 중...")
    print("=" * 60)

    train_result = trainer.train()

    trainer.save_model(os.path.join(config.path.output_dir, "checkpoint-last"))# (선택) 최종 모델 저장


    # 10. 결과 출력
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    print(f"  - Total steps: {train_result.global_step}")
    print(f"  - Train loss: {train_result.training_loss:.4f}")
    print(f"  - 체크포인트 저장 위치: {config.path.output_dir}")

    return trainer


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="실험 이름 (large_context, more_lora, longer_training)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="체크포인트 저장 경로"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="학습 에폭 수"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="학습률"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="최대 시퀀스 길이"
    )

    args = parser.parse_args()

    # 설정 로드
    if args.exp:
        config = get_experiment_config(args.exp)
        print(f"📌 실험 설정 로드: {args.exp}")
    else:
        config = get_config()
        print("📌 기본 설정 사용")

    # 명령줄 인자로 오버라이드
    if args.output_dir:
        config.path.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.max_seq_length:
        config.training.max_seq_length = args.max_seq_length

    # 학습 실행
    train(config)


if __name__ == "__main__":
    main()

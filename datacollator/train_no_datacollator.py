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
import evaluate

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from unsloth import FastLanguageModel, is_bfloat16_supported, UnslothTrainer, UnslothTrainingArguments

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


# =============================================================================
# 메트릭 함수
# =============================================================================
def create_metric_functions(tokenizer):
    import evaluate
    f1_metric = evaluate.load("f1")
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    # ✅ '1'~'5' 토큰 id를 안전하게 구하기 (tokenizer.vocab 의존 X)
    choice_token_ids = []
    for s in ["1", "2", "3", "4", "5"]:
        tid = tokenizer.encode(s, add_special_tokens=False)
        if len(tid) != 1:
            # 숫자가 한 토큰이 아닐 수도 있음. 이 경우는 별도 처리 필요.
            raise ValueError(f"토큰화가 예상과 달라요: '{s}' -> {tid}. 숫자만 1토큰이 아니면 로직을 바꿔야 해요.")
        choice_token_ids.append(tid[0])

    def preprocess_logits_for_metrics(logits, labels):
        """
        ✅ labels에서 정답 토큰 위치를 찾아서 그 위치의 logits만 뽑아온다.
        반환 shape: (batch, 5)  -> (1~5에 대한 점수)
        """
        logits = logits if not isinstance(logits, tuple) else logits[0]  # (B, T, V)
        B, T, V = logits.shape

        # labels: (B, T)
        labels_t = labels

        # 정답 위치 idx를 배치마다 구하기: labels != -100 인 위치 중 "첫 번째"
        # (지금은 답 토큰 + <|im_end|> 2개가 남아있으니 첫 번째가 답)
        ans_pos = []
        for i in range(B):
            positions = (labels_t[i] != -100).nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                # 혹시 전부 -100이면 안전하게 마지막 토큰으로(근데 이런 샘플은 metric에서 사실상 무의미)
                ans_pos.append(T - 1)
            else:
                ans_pos.append(int(positions[0].item()))
        ans_pos = torch.tensor(ans_pos, device=logits.device)  # (B,)

        # 배치 인덱싱으로 각 샘플의 정답 위치 logits을 뽑는다: (B, V)
        batch_idx = torch.arange(B, device=logits.device)
        ans_logits = logits[batch_idx, ans_pos, :]  # (B, V)

        # 그 중 1~5 토큰 id만 추출: (B, 5)
        ans_logits_5 = ans_logits[:, choice_token_ids]
        return ans_logits_5

    def compute_metrics(eval_pred):
        """
        preprocess_logits_for_metrics가 이미 (B,5) logits을 넘겨줌.
        labels에서 정답 숫자도 똑같이 labels 기반으로 추출해서 비교.
        """
        logits_5, labels = eval_pred  # logits_5: (B,5)

        # labels에서 정답 토큰 id 추출 (labels != -100의 첫 번째 토큰)
        B, T = labels.shape
        y_true = []
        for i in range(B):
            positions = np.where(labels[i] != -100)[0]
            if len(positions) == 0:
                y_true.append(0)
            else:
                tid = int(labels[i, positions[0]])
                # tid -> "1~5"로 매핑
                if tid in choice_token_ids:
                    y_true.append(choice_token_ids.index(tid))
                else:
                    # 예상 밖이면 0 처리
                    y_true.append(0)

        probs = torch.softmax(torch.tensor(logits_5), dim=-1)
        y_pred = torch.argmax(probs, dim=-1).cpu().numpy()

        return f1_metric.compute(predictions=y_pred, references=y_true, average="macro")

    return preprocess_logits_for_metrics, compute_metrics

# def create_metric_functions(tokenizer):
#     """메트릭 계산 함수들 생성"""

#     f1_metric = evaluate.load("f1")
#     int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

#     def preprocess_logits_for_metrics(logits, labels):
#         """정답 토큰 위치의 logits만 추출"""
#         logits = logits if not isinstance(logits, tuple) else logits[0]
#         logit_idx = [
#             tokenizer.vocab["1"],
#             tokenizer.vocab["2"],
#             tokenizer.vocab["3"],
#             tokenizer.vocab["4"],
#             tokenizer.vocab["5"]
#         ]
#         logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
#         return logits

#     def compute_metrics(evaluation_result):
#         """정확도 계산"""
#         logits, labels = evaluation_result

#         # 토큰화된 레이블 디코딩
#         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
#         labels = list(map(lambda x: int_output_map.get(x, 0), labels))

#         # Softmax로 확률 변환
#         probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
#         predictions = np.argmax(probs, axis=-1)

#         # 정확도 계산
#         macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
#         return macro_f1

#     return preprocess_logits_for_metrics, compute_metrics


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

    # 2. 데이터 로드 및 전처리
    print("\n📂 데이터 로드 중...")
    # df = load_data(config.path.train_data)
    # ✅ train / valid 데이터 둘 다 로드
    df_train = load_data(config.path.train_data)
    df_valid = load_data(config.path.valid_data)

    train_dataset_raw = process_dataset_for_training(df_train)
    valid_dataset_raw = process_dataset_for_training(df_valid)

    print(f"  - Train samples: {len(train_dataset_raw)}")
    print(f"  - Valid samples: {len(valid_dataset_raw)}")

    # print(f"  - 총 데이터 수: {len(df)}")

    # processed_dataset = process_dataset_for_training(df)
    # print(f"  - 전처리 완료: {len(processed_dataset)} samples")
    # print(f"💽 Data Format{processed_dataset['messages'][0:4]}")

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
        use_dora=config.lora.use_dora,

        init_lora_weights="pissa",

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

    sft_config = UnslothTrainingArguments(
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
        save_total_limit=config.training.save_total_limit,

        # ✅ best 모델 저장 (네가 추가한 3개가 여기로 들어와야 적용됨)
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
    )

    # 8. Trainer 생성
    trainer = UnslothTrainer(
        model=model,
        tokenizer = tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=sft_config,
    )

    # 9. 학습 실행
    print("\n" + "=" * 60)
    print("🏃 학습 실행 중...")
    print("=" * 60)

    train_result = trainer.train()

    print("best_ckpt:", trainer.state.best_model_checkpoint)
    print("best_metric:", trainer.state.best_metric)

    trainer.save_model(config.path.output_dir)  # (선택) 최종 모델 저장


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
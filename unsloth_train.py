"""
학습 스크립트
Usage:
    python train.py
    python train.py --exp large_context
"""
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
from unsloth import FastLanguageModel, is_bfloat16_supported

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
    """메트릭 계산 함수들 생성"""
    
    f1_metric = evaluate.load("f1")
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    
    def preprocess_logits_for_metrics(logits, labels):
        """정답 토큰 위치의 logits만 추출"""
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [
            tokenizer.vocab["1"], 
            tokenizer.vocab["2"], 
            tokenizer.vocab["3"], 
            tokenizer.vocab["4"], 
            tokenizer.vocab["5"]
        ]
        logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
        return logits

    def compute_metrics(evaluation_result):
        """정확도 계산"""
        logits, labels = evaluation_result

        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        labels = list(map(lambda x: int_output_map.get(x, 0), labels))

        # Softmax로 확률 변환
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 정확도 계산
        macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return macro_f1
    
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
    
    # 2. 데이터 로드 및 전처리
    print("\n📂 데이터 로드 중...")
    df = load_data(config.path.train_data)
    print(f"  - 총 데이터 수: {len(df)}")
    
    processed_dataset = process_dataset_for_training(df)
    print(f"  - 전처리 완료: {len(processed_dataset)} samples")
    
    # 3. 모델 및 토크나이저 로드
    print(f"\n🤖 모델 로드 중: {config.model.model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.model.model_name,
        dtype=get_torch_dtype(config.model.torch_dtype),
        # trust_remote_code=config.model.trust_remote_code,
        load_in_4bit=True
    )
    
    tokenizer = setup_tokenizer(tokenizer)  # Instruct 사용 시에는 필요없는 코드
    
    
    print("  - 모델 및 토크나이저 로드 완료")
    
    # 4. 토큰화
    print(f"\n📝 토큰화 중 (max_seq_length: {config.training.max_seq_length})...")
    tokenized_dataset = tokenize_dataset(
        processed_dataset, 
        tokenizer, 
        max_seq_length=config.training.max_seq_length
    )
    
    # Train/Eval 분할
    split_dataset = tokenized_dataset.train_test_split(
        test_size=config.training.test_size, 
        seed=config.training.seed
    )
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Eval: {len(eval_dataset)} samples")
    
    # 토큰 통계
    stats = get_token_statistics(train_dataset, tokenizer)
    print(f"  - Token 길이: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}")
    
    # 5. LoRA 설정
    print(f"\n⚙️ LoRA 설정")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing = "unsloth",
        random_state=config.training.seed,
        max_seq_length=config.training.max_seq_length,
        use_rslora=False,
        loftq_config=None
    )
    print(f"  - r: {config.lora.r}, alpha: {config.lora.lora_alpha}")
    print(f"  - target_modules: {config.lora.target_modules}")
    
    # 6. Data Collator 설정
    
    response_template = "<start_of_turn>model"
    
    """
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    """
    
    # 7. 메트릭 함수 생성
    preprocess_logits_for_metrics, compute_metrics = create_metric_functions(tokenizer)
    
    # 8. SFTConfig 설정
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
        # max_steps=config.training.max_steps,
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
    )
    
    # 9. Trainer 생성
    trainer = SFTTrainer(
        model=model,
        tokenizer = tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=sft_config,
    )
    
    # 10. 학습 실행
    print("\n" + "=" * 60)
    print("🏃 학습 실행 중...")
    print("=" * 60)
    
    train_result = trainer.train()
    
    # 11. 결과 출력
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
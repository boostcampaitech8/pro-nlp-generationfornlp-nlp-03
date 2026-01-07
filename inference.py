"""
Inference 스크립트
Usage:
    # Valid set으로 F1/Accuracy 계산
    python inference.py --mode valid --checkpoint ./results/checkpoint-best

    # Test set으로 submission.csv 생성
    python inference.py --mode test --checkpoint ./results/checkpoint-best
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from peft import PeftModel  # ✅ 추가

from config import get_config
from data_utils import (
    load_data,
    process_dataset_for_inference,
    setup_tokenizer,
)


def get_answer_token_ids(tokenizer):
    """1~5 토큰 ID 추출"""
    choice_token_ids = []
    for s in ["1", "2", "3", "4", "5"]:
        tid = tokenizer.encode(s, add_special_tokens=False)
        if len(tid) != 1:
            raise ValueError(f"토큰화 예상과 다름: '{s}' -> {tid}")
        choice_token_ids.append(tid[0])

    print(f"✓ 정답 토큰 ID: {dict(zip(['1','2','3','4','5'], choice_token_ids))}")
    return choice_token_ids


def inference_with_logits(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    choice_token_ids,
):
    """
    Logits 기반 추론 (train.py metric과 동일 방식)

    Returns:
        predicted answer (0~4)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    # ✅ assistant 답변 시작 위치 찾기
    # <|im_start|>assistant 다음 토큰이 정답 (1~5)
    response_template = "<|im_start|>assistant"
    response_ids = tokenizer.encode(response_template, add_special_tokens=False)

    batch_size = input_ids.size(0)
    predictions = []

    for i in range(batch_size):
        ids_list = input_ids[i].tolist()

        # response_template 위치 찾기
        start = _find_sublist(ids_list, response_ids)
        if start == -1:
            # 못 찾으면 마지막 토큰 사용
            ans_pos = len(ids_list) - 1
        else:
            # response_template 끝 다음 = 정답 토큰 위치
            ans_pos = start + len(response_ids)

        # 해당 위치의 logits에서 1~5 토큰만 추출
        ans_logits = logits[i, ans_pos, choice_token_ids]  # (5,)
        pred = torch.argmax(ans_logits).item()  # 0~4
        predictions.append(pred)

    return predictions


def _find_sublist(haystack, needle):
    """리스트에서 서브리스트 찾기"""
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1


def run_inference_valid(config, checkpoint_path):
    """Valid set으로 F1/Accuracy 계산"""

    print("=" * 60)
    print("📊 Valid Set Inference (F1 + Accuracy)")
    print("=" * 60)

    # 1. 모델/토크나이저 로드
    print(f"\n🤖 모델 로드 중...")
    print(f"  - Base model: {config.model.model_name}")
    print(f"  - Adapter: {checkpoint_path}")

    # ✅ Base model 먼저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.model.model_name,  # ✅ Base model
        dtype=torch.float16,
        load_in_4bit=True,
        max_seq_length=config.training.max_seq_length,
    )

    # ✅ Adapter 로드
    # from peft import PeftModel  # Already imported at top
    model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    tokenizer = setup_tokenizer(tokenizer)

    print("  ✅ 모델 로드 완료")

    choice_token_ids = get_answer_token_ids(tokenizer)

    # 2. Valid 데이터 로드
    print(f"\n📂 Valid 데이터 로드: {config.path.valid_data}")
    df_valid = load_data(config.path.valid_data)
    valid_data = process_dataset_for_inference(df_valid)
    print(f"  - Valid samples: {len(valid_data)}")

    # ✅ Label 체크
    none_labels = [d['id'] for d in valid_data if d['label'] is None]
    if none_labels:
        print(f"\n⚠️ 경고: {len(none_labels)}개 샘플에 answer가 없습니다!")
        print(f"  처음 5개: {none_labels[:5]}")
        print(f"  → Valid set은 answer가 필요합니다. 데이터를 확인하세요.")
        # None 제거
        valid_data = [d for d in valid_data if d['label'] is not None]
        print(f"  → 필터링 후: {len(valid_data)} samples")

    # 3. Inference
    print("\n🔮 Inference 실행 중...")
    all_predictions = []
    all_labels = []

    device = model.device
    batch_size = config.training.per_device_eval_batch_size

    for i in tqdm(range(0, len(valid_data), batch_size), desc="Processing"):
        batch_data = valid_data[i:i+batch_size]

        # 토큰화
        batch_texts = []
        batch_labels = []
        for item in batch_data:
            # ✅ label이 None인 경우 체크
            if item["label"] is None:
                print(f"\n⚠️ 경고: {item['id']} - label이 None입니다. 스킵합니다.")
                continue

            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_texts.append(text)
            batch_labels.append(item["label"] - 1)  # 1~5 -> 0~4

        # ✅ batch가 비어있으면 스킵
        if len(batch_texts) == 0:
            continue

        # 인코딩
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.training.max_seq_length,
        ).to(device)

        # 예측
        predictions = inference_with_logits(
            model,
            tokenizer,
            encoded["input_ids"],
            encoded["attention_mask"],
            choice_token_ids,
        )

        all_predictions.extend(predictions)
        all_labels.extend(batch_labels)

    # 4. 메트릭 계산
    print("\n" + "=" * 60)
    print("✅ 결과")
    print("=" * 60)

    f1 = f1_score(all_labels, all_predictions, average="macro")
    acc = accuracy_score(all_labels, all_predictions)

    print(f"  - Macro F1:  {f1:.4f}")
    print(f"  - Accuracy:  {acc:.4f}")
    print(f"  - Total samples: {len(all_labels)}")

    # 5. 오답 분석 (선택)
    errors = []
    for i, (pred, label) in enumerate(zip(all_predictions, all_labels)):
        if pred != label:
            errors.append({
                'id': valid_data[i]['id'],
                'predicted': pred + 1,
                'actual': label + 1,
            })

    if errors:
        print(f"\n❌ 오답: {len(errors)}개")
        print("  처음 5개:")
        for err in errors[:5]:
            print(f"    {err['id']}: 예측={err['predicted']}, 정답={err['actual']}")

    return f1, acc


def run_inference_test(config, checkpoint_path):
    """Test set으로 submission.csv 생성"""

    print("=" * 60)
    print("📝 Test Set Inference (Submission 생성)")
    print("=" * 60)

    # 1. 모델/토크나이저 로드
    print(f"\n🤖 모델 로드 중...")
    print(f"  - Base model: {config.model.model_name}")
    print(f"  - Adapter: {checkpoint_path}")

    # ✅ Base model 먼저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.model.model_name,  # ✅ Base model
        dtype=torch.float16,
        load_in_4bit=True,
        max_seq_length=config.training.max_seq_length,
    )

    # ✅ Adapter 로드
    # from peft import PeftModel  # Already imported at top
    model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    tokenizer = setup_tokenizer(tokenizer)

    print("  ✅ 모델 로드 완료")

    choice_token_ids = get_answer_token_ids(tokenizer)

    # 2. Test 데이터 로드
    print(f"\n📂 Test 데이터 로드: {config.path.test_data}")
    df_test = load_data(config.path.test_data)
    test_data = process_dataset_for_inference(df_test)
    print(f"  - Test samples: {len(test_data)}")

    # 3. Inference
    print("\n🔮 Inference 실행 중...")
    all_predictions = []
    all_ids = []

    device = model.device
    batch_size = config.training.per_device_eval_batch_size

    for i in tqdm(range(0, len(test_data), batch_size), desc="Processing"):
        batch_data = test_data[i:i+batch_size]

        # 토큰화
        batch_texts = []
        batch_ids = []
        for item in batch_data:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_texts.append(text)
            batch_ids.append(item["id"])

        # 인코딩
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.training.max_seq_length,
        ).to(device)

        # 예측
        predictions = inference_with_logits(
            model,
            tokenizer,
            encoded["input_ids"],
            encoded["attention_mask"],
            choice_token_ids,
        )

        # 0~4 -> 1~5로 변환
        predictions = [p + 1 for p in predictions]

        all_predictions.extend(predictions)
        all_ids.extend(batch_ids)

    # 4. Submission CSV 생성
    print("\n💾 Submission 저장 중...")
    submission_df = pd.DataFrame({
        'id': all_ids,
        'answer': all_predictions,
    })

    # output_csv 경로 사용
    output_path = config.path.output_csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"  ✅ 저장 완료: {output_path}")
    print(f"  - Total: {len(submission_df)} rows")

    # 5. 샘플 출력
    print("\n📋 Submission 샘플 (처음 10개):")
    print(submission_df.head(10).to_string(index=False))

    # 6. 답변 분포
    print("\n📊 답변 분포:")
    for ans in [1, 2, 3, 4, 5]:
        count = (submission_df['answer'] == ans).sum()
        pct = count / len(submission_df) * 100
        print(f"  {ans}: {count} ({pct:.1f}%)")

    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["valid", "test"],
        help="valid: F1/Acc 계산, test: submission.csv 생성"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="체크포인트 경로 (예: ./results/checkpoint-best)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="커스텀 config 파일 경로 (선택)"
    )

    args = parser.parse_args()

    # Config 로드
    config = get_config()
    print(f"📌 Config 로드 완료")
    print(f"  - Max seq length: {config.training.max_seq_length}")
    print(f"  - Eval batch size: {config.training.per_device_eval_batch_size}")

    # Inference 실행
    if args.mode == "valid":
        f1, acc = run_inference_valid(config, args.checkpoint)
        print("\n" + "=" * 60)
        print("🎉 Valid Inference 완료!")
        print("=" * 60)

    elif args.mode == "test":
        submission_df = run_inference_test(config, args.checkpoint)
        print("\n" + "=" * 60)
        print("🎉 Test Inference 완료!")
        print(f"📁 Submission: {config.path.output_csv}")
        print("=" * 60)


if __name__ == "__main__":
    main()
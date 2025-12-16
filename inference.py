"""
추론 스크립트
Usage:
    python inference.py --checkpoint outputs/checkpoint-4491
    python inference.py --checkpoint outputs/checkpoint-4491 --output my_output.csv
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

from config import get_config
from data_utils import load_data, process_dataset_for_inference, setup_tokenizer


# =============================================================================
# 추론 함수
# =============================================================================

def inference(
    checkpoint_path: str,
    test_data_path: str,
    output_path: str,
    device: str = "cuda"
):
    """
    모델 추론 실행
    
    Args:
        checkpoint_path: 학습된 체크포인트 경로
        test_data_path: 테스트 데이터 경로
        output_path: 결과 저장 경로
        device: 디바이스 ("cuda" or "cpu")
    """
    
    print("=" * 60)
    print("🔮 추론 시작")
    print("=" * 60)
    
    # 1. 모델 및 토크나이저 로드
    print(f"\n🤖 모델 로드 중: {checkpoint_path}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    tokenizer = setup_tokenizer(tokenizer)
    print("  - 모델 및 토크나이저 로드 완료")
    
    # 2. 테스트 데이터 로드
    print(f"\n📂 테스트 데이터 로드 중: {test_data_path}")
    test_df = load_data(test_data_path)
    test_dataset = process_dataset_for_inference(test_df)
    print(f"  - 테스트 샘플 수: {len(test_dataset)}")
    
    # 3. 추론 실행
    print("\n" + "=" * 60)
    print("🏃 추론 실행 중...")
    print("=" * 60)
    
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset, desc="Inference"):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]
            
            # 토큰화
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            
            # 모델 추론
            outputs = model(inputs)
            logits = outputs.logits[:, -1].flatten().cpu()
            
            # 선택지 토큰의 logit 추출
            target_logit_list = [
                logits[tokenizer.vocab[str(i + 1)]] 
                for i in range(len_choices)
            ]
            
            # Softmax로 확률 변환
            probs = torch.nn.functional.softmax(
                torch.tensor(target_logit_list, dtype=torch.float32),
                dim=-1
            ).detach().cpu().numpy()
            
            # 최종 예측
            predict_value = pred_choices_map[np.argmax(probs)]
            infer_results.append({"id": _id, "answer": predict_value})
    
    # 4. 결과 저장
    print(f"\n💾 결과 저장 중: {output_path}")
    result_df = pd.DataFrame(infer_results)
    result_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("✅ 추론 완료!")
    print("=" * 60)
    print(f"  - 총 예측 수: {len(infer_results)}")
    print(f"  - 저장 위치: {output_path}")
    
    # 예측 분포 출력
    print("\n📊 예측 분포:")
    value_counts = result_df['answer'].value_counts().sort_index()
    for ans, count in value_counts.items():
        print(f"  - {ans}: {count} ({count/len(result_df)*100:.1f}%)")
    
    return result_df


def inference_with_generation(
    checkpoint_path: str,
    test_data_path: str,
    output_path: str,
    device: str = "cuda",
    max_new_tokens: int = 5
):
    """
    생성 방식 추론 (generate 함수 사용)
    - logit 기반보다 느리지만, 더 복잡한 출력에 적합
    """
    
    print("=" * 60)
    print("🔮 추론 시작 (Generation Mode)")
    print("=" * 60)
    
    # 모델 로드
    print(f"\n🤖 모델 로드 중: {checkpoint_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    tokenizer = setup_tokenizer(tokenizer)
    
    # 테스트 데이터 로드
    print(f"\n📂 테스트 데이터 로드 중: {test_data_path}")
    test_df = load_data(test_data_path)
    test_dataset = process_dataset_for_inference(test_df)
    print(f"  - 테스트 샘플 수: {len(test_dataset)}")
    
    # 추론
    print("\n🏃 추론 실행 중...")
    infer_results = []
    
    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset, desc="Inference"):
            _id = data["id"]
            messages = data["messages"]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            
            # 생성
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # 디코딩 (생성된 부분만)
            generated = outputs[0][inputs.shape[1]:]
            answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            # 정답 추출 (첫 번째 숫자)
            answer = "1"  # 기본값
            for char in answer_text:
                if char in "12345":
                    answer = char
                    break
            
            infer_results.append({"id": _id, "answer": answer})
    
    # 결과 저장
    result_df = pd.DataFrame(infer_results)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✅ 추론 완료! 저장 위치: {output_path}")
    
    return result_df


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="학습된 체크포인트 경로"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="테스트 데이터 경로 (기본: config의 test_data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로 (기본: config의 output_csv)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="logit",
        choices=["logit", "generate"],
        help="추론 방식 (logit: logit 기반, generate: 생성 기반)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="디바이스 (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = get_config()
    
    test_data_path = args.test_data or config.path.test_data
    output_path = args.output or config.path.output_csv
    
    # 추론 실행
    if args.mode == "logit":
        inference(
            checkpoint_path=args.checkpoint,
            test_data_path=test_data_path,
            output_path=output_path,
            device=args.device,
        )
    else:
        inference_with_generation(
            checkpoint_path=args.checkpoint,
            test_data_path=test_data_path,
            output_path=output_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()

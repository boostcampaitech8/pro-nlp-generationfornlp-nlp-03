"""
inference_rag.py
추론 스크립트 (RAG 지원)
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config_first import get_config
from data_utils_rag import load_data, process_dataset_for_inference, setup_tokenizer


def inference(
    checkpoint_path: str,
    test_data_path: str,
    output_path: str,
    device: str = "cuda",
    use_4bit: bool = True,
    use_rag: bool = False 
):
    """
    Logit 기반 추론
    
    Args:
        use_rag: external_facts 사용 여부
    """
    
    print("=" * 60)
    print(f"추론 시작 (Logit 모드, RAG={'ON' if use_rag else 'OFF'})")
    print("=" * 60)
    
    # 모델 로드
    print(f"\n모델 로드 (4bit): {checkpoint_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    tokenizer = setup_tokenizer(tokenizer)
    print("모델 로드 완료")
    
    # 정답 토큰 ID
    vocab = tokenizer.get_vocab()
    answer_token_ids = {
        "1": vocab.get("1"),
        "2": vocab.get("2"),
        "3": vocab.get("3"),
        "4": vocab.get("4"),
        "5": vocab.get("5"),
    }
    
    print("\n정답 토큰 ID:")
    for num, token_id in answer_token_ids.items():
        print(f"  '{num}' -> {token_id}")
    
    # 테스트 데이터
    print(f"\n테스트 데이터: {test_data_path}")
    test_df = load_data(test_data_path)
    
    # RAG 사용 여부 체크
    if use_rag:
        if 'external_facts' not in test_df.columns:
            print("  WARNING: use_rag=True but 'external_facts' column not found!")
            print("    Proceeding without RAG...")
            use_rag = False
        else:
            print(f"RAG enabled: external_facts column found")
    
    test_dataset = process_dataset_for_inference(test_df, use_rag=use_rag)
    print(f"총 샘플: {len(test_dataset)}")
    
    # 추론
    print("\n추론 중...")
    infer_results = []
    pred_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            inputs = tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            
            outputs = model(inputs)
            logits = outputs.logits[:, -1].flatten().cpu()
            
            # 선택지 토큰 logit 추출
            target_logits = []
            for i in range(data["len_choices"]):
                num_str = str(i + 1)
                token_id = answer_token_ids.get(num_str)
                
                if token_id is not None:
                    target_logits.append(logits[token_id].item())
                else:
                    target_logits.append(-100.0)
            
            # Softmax
            probs = torch.nn.functional.softmax(
                torch.tensor(target_logits),
                dim=-1
            ).numpy()
            
            predict = pred_map[np.argmax(probs)]
            infer_results.append({"id": data["id"], "answer": predict})
    
    # 저장
    result_df = pd.DataFrame(infer_results)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n완료! 저장: {output_path}")
    print(f"총 예측: {len(infer_results)}")
    
    # 분포 확인
    print("\n예측 분포:")
    value_counts = result_df['answer'].value_counts().sort_index()
    for ans, count in value_counts.items():
        print(f"  {ans}: {count} ({count/len(result_df)*100:.1f}%)")
    
    return result_df

def inference_generate(
    checkpoint_path: str,
    test_data_path: str,
    output_path: str,
    device: str = "cuda",
    use_rag: bool = False  # 추가!
):
    """
    Generate 기반 추론 (RAG 지원)
    """
    
    print("=" * 60)
    print(f"추론 시작 (Generate 모드, RAG={'ON' if use_rag else 'OFF'})")
    print("=" * 60)
    
    # 모델 로드
    print(f"\n모델 로드: {checkpoint_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    tokenizer = setup_tokenizer(tokenizer)
    print("모델 로드 완료")
    
    # 데이터
    print(f"\n데이터 로드: {test_data_path}")
    test_df = load_data(test_data_path)
    
    # RAG 체크
    if use_rag:
        if 'external_facts' not in test_df.columns:
            print("  WARNING: use_rag=True but 'external_facts' not found!")
            use_rag = False
        else:
            print(f" RAG enabled")
    
    test_dataset = process_dataset_for_inference(test_df, use_rag=use_rag)
    print(f"총 샘플: {len(test_dataset)}")
    
    # 추론
    print("\n추론 중...")
    results = []
    
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            # 토큰화
            inputs = tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            
            input_length = inputs.shape[1]
            
            # Generate
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_ids = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 숫자 추출 (더 강건하게)
            answer = "1"  # 기본값
            
            # 방법 1: 첫 번째 숫자 찾기
            for char in generated_text:
                if char in "12345":
                    answer = char
                    break
            
            # 방법 2: "정답은 X" 패턴 찾기
            import re
            match = re.search(r'[정답은\s]*([1-5])', generated_text)
            if match:
                answer = match.group(1)
            
            results.append({"id": data["id"], "answer": answer})
    
    # 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n완료! 저장: {output_path}")
    print(f"총 예측: {len(results)}")
    
    # 분포
    print("\n예측 분포:")
    value_counts = result_df['answer'].value_counts().sort_index()
    for ans, count in value_counts.items():
        print(f"  {ans}: {count} ({count/len(result_df)*100:.1f}%)")
    
    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", type=str, default="generate",  # 기본값 변경
                       choices=["logit", "generate"])
    parser.add_argument("--use_rag", action="store_true",
                       help="Use external_facts from RAG")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    config = get_config()
    test_data = args.test_data or config.path.test_data
    output = args.output or config.path.output_csv
    
    if args.mode == "generate":
        inference_generate(
            args.checkpoint, 
            test_data, 
            output, 
            args.device,
            use_rag=args.use_rag
        )
    elif args.mode == "logit":
        inference(
            args.checkpoint, 
            test_data, 
            output, 
            args.device,
            use_rag=args.use_rag
        )

if __name__ == "__main__":
    main()
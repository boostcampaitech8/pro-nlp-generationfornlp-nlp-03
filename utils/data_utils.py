"""
데이터 처리 모듈
데이터 로드, 전처리, 토큰화 담당
"""

import pandas as pd
from ast import literal_eval
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from utils.prompt_utils import (
        COT_PROMPT_NO_QUESTION_PLUS, 
        COT_PROMPT_QUESTION_PLUS, 
        COT_SYSTEM_MESSAGE,
        PROMPT_NO_QUESTION_PLUS,
        PROMPT_QUESTION_PLUS,
        SYSTEM_MESSAGE
    )


# =============================================================================
# 프롬프트 템플릿
# =============================================================================

PROMPT_NO_QUESTION_PLUS = PROMPT_NO_QUESTION_PLUS

PROMPT_QUESTION_PLUS = PROMPT_QUESTION_PLUS

SYSTEM_MESSAGE = SYSTEM_MESSAGE

# Gemma용 Chat Template
CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\\n' }}{% endif %}{% endfor %}"


# =============================================================================
# 데이터 로드 함수
# =============================================================================

def parse_choices(x):
    if isinstance(x, list):
        return x

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return literal_eval(s)
            except Exception:
                # 깨진 리스트 문자열
                return []

    return []

def load_data(file_path: str) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 problems 컬럼을 파싱
    
    Args:
        file_path: CSV 파일 경로
        
    Returns:
        파싱된 DataFrame
    """
    dataset = pd.read_csv(file_path)
    
    records = []
    
    if 'question' not in dataset.columns:
        for _, row in dataset.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                'question_plus': problems.get('question_plus', None),
                'topic': row.get('topic', None),
            }
            records.append(record)
    else:
        for _, row in dataset.iterrows():
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': row['question'],
                'choices': parse_choices(row['choices']),
                'answer': row.get('answer', None),
                'question_plus': row.get('question_plus', None),
                'topic': row.get('topic', None),
                'type': row['type'],
                'stratify_key': row['stratify_key']
            }
            records.append(record)
    
    return pd.DataFrame(records)


# =============================================================================
# 전처리 함수
# =============================================================================

def format_choices(choices: List[str]) -> str:
    """선택지를 문자열로 포맷팅"""
    return "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])


def create_user_message(row: Dict) -> str:
    """데이터 행에서 user 메시지 생성"""
    choices_string = format_choices(row["choices"])
    
    if row["question_plus"]:
        return PROMPT_QUESTION_PLUS.format(
            paragraph=row["paragraph"],
            question=row["question"],
            question_plus=row["question_plus"],
            choice_count=len(row["choices"]),
            choices=choices_string,
        )
    else:
        return PROMPT_NO_QUESTION_PLUS.format(
            paragraph=row["paragraph"],
            question=row["question"],
            choice_count=len(row["choices"]),
            choices=choices_string,
        )


def process_dataset_for_training(df: pd.DataFrame) -> Dataset:
    """
    학습용 데이터셋 전처리
    
    Args:
        df: 원본 DataFrame
        
    Returns:
        전처리된 HuggingFace Dataset
    """
    processed_data = []
    
    for _, row in df.iterrows():
        user_message = create_user_message(row)
        
        processed_data.append({
            "id": row["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": str(row["answer"])}
            ],
            "label": row["answer"],
        })
    
    return Dataset.from_pandas(pd.DataFrame(processed_data))


def process_dataset_for_inference(df: pd.DataFrame) -> List[Dict]:
    """
    추론용 데이터셋 전처리
    
    Args:
        df: 원본 DataFrame
        
    Returns:
        전처리된 데이터 리스트
    """
    processed_data = []
    
    for _, row in df.iterrows():
        user_message = create_user_message(row)
        
        processed_data.append({
            "id": row["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            "label": row.get("answer"),  # test 데이터에는 없을 수 있음
            "topic": row.get('topic'),
            "type": row.get("type"),
            "stratify_key": row.get('stratify_key'),
            "len_choices": len(row["choices"]),
        })
    
    return processed_data


# =============================================================================
# 토큰화 함수
# =============================================================================

def setup_tokenizer(tokenizer):
    """토크나이저 설정"""
    # Chat template 설정 (없는 경우)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE
    
    # Pad token 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    
    return tokenizer


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int = 1024,
    num_proc: int = 4
) -> Tuple[Dataset, Dataset]:
    """
    데이터셋 토큰화 및 train/eval 분할
    
    Args:
        dataset: 전처리된 Dataset
        tokenizer: 토크나이저
        max_seq_length: 최대 시퀀스 길이
        num_proc: 병렬 처리 수
        
    Returns:
        (train_dataset, eval_dataset) 튜플
    """
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts

    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    # 토큰화
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    
    # 길이 필터링
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) <= max_seq_length,
        desc="Filtering by length"
    )
    
    return tokenized_dataset


def get_token_statistics(dataset: Dataset, tokenizer) -> Dict:
    """토큰 길이 통계 계산"""
    lengths = [len(dataset[i]["input_ids"]) for i in range(len(dataset))]
    
    return {
        "max": max(lengths),
        "min": min(lengths),
        "mean": sum(lengths) / len(lengths),
        "count": len(lengths),
    }

"""
data_util_rag.py
데이터 처리 모듈
데이터 로드, 전처리, 토큰화 담당
"""

import pandas as pd
from ast import literal_eval
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
import json


# =============================================================================
# 프롬프트 템플릿
# =============================================================================

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

# RAG 포함 프롬프트 (새로 추가)
PROMPT_WITH_RAG = """지문:
{paragraph}

질문:
{question}
{external_facts}
선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_WITH_RAG_AND_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}
{external_facts}
선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

SYSTEM_MESSAGE = "지문을 읽고 질문의 답을 구하세요."

# Qwen용 Chat Template
CHAT_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


# =============================================================================
# 데이터 로드 함수
# =============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 다양한 형식을 통합된 DataFrame으로 변환
    """
    dataset = pd.read_csv(file_path)
    records = []
    
    for _, row in dataset.iterrows():
        # Case 1: 기존 train.csv 형식 (problems 컬럼이 있는 경우)
        if 'problems' in dataset.columns and pd.notna(row['problems']):
            try:
                problems = row['problems']
                if isinstance(problems, str):
                    problems = literal_eval(problems)
                
                record = {
                    'id': row['id'],
                    'paragraph': row['paragraph'],
                    'question': problems['question'],
                    'choices': problems['choices'],
                    'answer': problems.get('answer', None),
                    'question_plus': row.get('question_plus', problems.get('question_plus', None)),
                    'external_facts': row.get('external_facts', None),  # 추가
                }
            except:
                continue
                
        # Case 2: train_preprocessed.csv 형식
        else:
            choices = row['choices']
            
            if isinstance(choices, str):
                choices = choices.strip()
                
                try:
                    choices = literal_eval(choices)
                except (SyntaxError, ValueError):
                    import json
                    try:
                        choices = json.loads(choices.replace("'", '"'))
                    except:
                        choices = [c.strip().strip('"').strip("'") for c in choices.strip("[]").split('", "')]

            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': row['question'],
                'choices': choices,
                'answer': row.get('answer', None),
                'question_plus': row.get('question_plus', None),
                'external_facts': row.get('external_facts', None),  # 추가
            }
        
        records.append(record)
    
    return pd.DataFrame(records)


# =============================================================================
# 전처리 함수
# =============================================================================

def format_choices(choices: List[str]) -> str:
    """선택지를 문자열로 포맷팅"""
    return "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])


def format_external_facts(
    external_facts_json: Optional[str], 
    max_facts: int = 10,  
    max_fact_length: int = None  # 자르지 않음
) -> str:
    """
    External facts를 프롬프트 형식으로 포맷팅
    
    Args:
        external_facts_json: JSON 문자열 형태의 external_facts
        max_facts: 최대 사실 개수
        max_fact_length: 각 fact의 최대 길이 (None이면 자르지 않음)
        
    Returns:
        포맷팅된 문자열
    """
    if not external_facts_json or pd.isna(external_facts_json):
        return ""
    
    try:
        # JSON 파싱
        facts_data = json.loads(external_facts_json)
        
        target = facts_data.get('target', '')
        facts = facts_data.get('facts', [])
        sources = facts_data.get('sources', [])
        
        if not target and not facts:
            return ""
        
        # 포맷팅
        result = "\n\n### 참고 자료\n"
        
        if target:
            result += f"**대상**: {target}\n\n"
        
        if facts:
            result += "**관련 사실**:\n"
            for i, fact in enumerate(facts[:max_facts], 1):
                # max_fact_length가 있으면 자르기, 없으면 그대로
                if max_fact_length and len(fact) > max_fact_length:
                    fact_text = fact[:max_fact_length] + "..."
                else:
                    fact_text = fact
                
                # 보기 정보 추가 (있으면)
                related_info = ""
                if i <= len(sources):
                    source = sources[i-1]
                    if 'related_to' in source:
                        related_info = f" {source['related_to']}"
                
                result += f"{i}.{related_info} {fact_text}\n"
        
        result += "\n"
        return result
        
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Warning: Failed to parse external_facts: {e}")
        return ""


def create_user_message(row: Dict, use_rag: bool = False) -> str:
    """
    데이터 행에서 user 메시지 생성
    
    Args:
        row: 데이터 행
        use_rag: RAG facts 사용 여부
        
    Returns:
        포맷팅된 user 메시지
    """
    choices_string = format_choices(row["choices"])
    
    # RAG 사용
    if use_rag:
        external_facts = format_external_facts(row.get("external_facts"))
        
        if row.get("question_plus"):
            return PROMPT_WITH_RAG_AND_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                external_facts=external_facts,
                choices=choices_string,
            )
        else:
            return PROMPT_WITH_RAG.format(
                paragraph=row["paragraph"],
                question=row["question"],
                external_facts=external_facts,
                choices=choices_string,
            )
    
    # 기존 방식 (RAG 없음)
    else:
        if row.get("question_plus"):
            return PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        else:
            return PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )


def process_dataset_for_training(df: pd.DataFrame, use_rag: bool = False) -> Dataset:
    """
    학습용 데이터셋 전처리
    
    Args:
        df: 원본 DataFrame
        use_rag: RAG facts 사용 여부
        
    Returns:
        전처리된 HuggingFace Dataset
    """
    processed_data = []
    
    for _, row in df.iterrows():
        user_message = create_user_message(row, use_rag=use_rag)
        
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


def process_dataset_for_inference(df: pd.DataFrame, use_rag: bool = False) -> List[Dict]:
    """
    추론용 데이터셋 전처리
    
    Args:
        df: 원본 DataFrame
        use_rag: RAG facts 사용 여부
        
    Returns:
        전처리된 데이터 리스트
    """
    processed_data = []
    
    for _, row in df.iterrows():
        user_message = create_user_message(row, use_rag=use_rag)
        
        processed_data.append({
            "id": row["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            "label": row.get("answer"),
            "len_choices": len(row["choices"]),
        })
    
    return processed_data


# =============================================================================
# 토큰화 함수
# =============================================================================

def setup_tokenizer(tokenizer):
    """토크나이저 설정"""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE
    
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

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    
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
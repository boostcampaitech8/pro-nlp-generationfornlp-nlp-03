from ast import literal_eval
import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate

from typing import Dict
from datasets import Dataset

sns.set_theme(style="whitegrid")

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

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


def get_token_statistics(dataset: Dataset, tokenizer) -> Dict:
    """토큰 길이 통계 계산"""
    lengths = [len(dataset[i]["input_ids"]) for i in range(len(dataset))]

    return {
        "max": max(lengths),
        "min": min(lengths),
        "mean": sum(lengths) / len(lengths),
        "count": len(lengths),
    }


# =============================================================================
# 성능 평가 함수
# =============================================================================


def create_metric_functions(tokenizer):

    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    f1_macro = evaluate.load("f1")
    acc = evaluate.load("accuracy")

    """메트릭 계산 함수들 생성"""

    def preprocess_logits_for_metrics(logits, labels):
        """정답 토큰 위치의 logits만 추출"""
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [
            tokenizer.vocab["1"],
            tokenizer.vocab["2"],
            tokenizer.vocab["3"],
            tokenizer.vocab["4"],
            tokenizer.vocab["5"],
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
        macro_f1 = f1_macro.compute(
            predictions=predictions, references=labels, average="macro"
        )
        acc = acc.compute(predictions=predictions, references=labels)

        return {"macro_f1": macro_f1, "accuracy": acc}

    return preprocess_logits_for_metrics, compute_metrics


# =============================================================================
# 시각화 함수
# =============================================================================


def analyze_subject_accuracy(
    df,
    true_col="label",  # 진짜 정답
    pred_col="answer",  # 모델 예측
    topic_col="topic",  # 과목 (사회, 경제, …)
    save_dir=None,
):
    """
    [보고용]
    과목별 정답 개수 / 전체 개수 / 정답 비율 분석

    기준:
    - 정답 여부: true_col == pred_col
    - 과목별 집계: topic_col
    """

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    df = df.copy()

    # -----------------------------
    # 정답 여부 판단 (핵심)
    # -----------------------------
    df["correct"] = df[true_col] == df[pred_col]

    # -----------------------------
    # 과목별 집계
    # -----------------------------
    result_df = (
        df.groupby(topic_col)
        .agg(total_count=("correct", "size"), correct_count=("correct", "sum"))
        .reset_index()
    )

    result_df["correct_ratio"] = result_df["correct_count"] / result_df["total_count"]

    # 정답률 낮은 과목부터 정렬
    result_df = result_df.sort_values("correct_ratio")

    # -----------------------------
    # 표 출력 (보고용)
    # -----------------------------
    print(f"\n📊 {topic_col}-wise Accuracy Report")
    print(
        result_df.rename(
            columns={
                topic_col: "Subject",
                "total_count": "Total",
                "correct_count": "Correct",
                "correct_ratio": "Accuracy",
            }
        )
    )

    # -----------------------------
    # 시각화 (보고용)
    # -----------------------------
    plt.figure(figsize=(11, max(4, len(result_df) * 0.38)))

    colors = [
        "firebrick" if r < 0.3 else "darkorange" if r < 0.6 else "seagreen"
        for r in result_df["correct_ratio"]
    ]

    bars = plt.barh(result_df[topic_col], result_df["correct_ratio"], color=colors)

    for _, row in result_df.iterrows():
        plt.text(
            row["correct_ratio"] + 0.01,
            row[topic_col],
            f"{row['correct_count']} / {row['total_count']}  ({row['correct_ratio']:.2f})",
            va="center",
            fontsize=10,
        )

    plt.xlim(0, 1)
    plt.xlabel("Accuracy (Correct / Total)")
    plt.title(
        f"{topic_col}-wise Accuracy\n(How many questions were answered correctly per {topic_col})",
        fontsize=14,
        weight="bold",
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📂 폴더 생성 완료: {save_dir}")

    if save_dir:
        plt.savefig(f"{save_dir}/{topic_col}_accuracy.png")

    plt.show()

    return result_df


def balance_answer_by_swap(df):
    """
    Using the 'choice_len' column,
    evenly distribute the correct answers in the data.
    """

    invalid_mask = (df["choice_len"] <= 0) | (df["choice_len"] > 5)

    if invalid_mask.any():
        invalid_count = len(df[invalid_mask])
        print(
            f"⚠️ 경고: 유효하지 않은 선택지 개수(0개 또는 5개 초과)를 가진 데이터 {invalid_count}건을 제외합니다."
        )
        # 유효한 범위(1~5)인 데이터만 남김
        df = df[~invalid_mask].reset_index(drop=True)

    total_len = len(df)

    # 1. 전체 목표 정답 리스트 생성 (1~5번이 각 812개씩)
    all_targets = ([1, 2, 3, 4, 5] * (total_len // 5 + 1))[:total_len]
    np.random.seed(42)
    np.random.shuffle(all_targets)

    # 제약 조건 해결 (4지선다는 5번을 가질 수 없음)
    # 목표 정답이 5인데 문항이 4지선다인 경우, 5지선다 문항의 1~4번 정답과 맞바꿈
    df_4 = df[df["choice_len"] == 4].index.tolist()
    df_5 = df[df["choice_len"] == 5].index.tolist()

    # 5번 정답이 배정된 인덱스들
    target_5_indices = [i for i, val in enumerate(all_targets) if val == 5]

    for idx in target_5_indices:
        # 만약 5번 정답이 배정된 곳이 4지선다 문항이라면?
        if idx in df_4:
            # 5지선다 문항 중 정답이 1~4번으로 배정된 아무 인덱스나 찾아서 교체
            for swap_idx in range(total_len):
                if swap_idx in df_5 and all_targets[swap_idx] != 5:
                    all_targets[idx], all_targets[swap_idx] = (
                        all_targets[swap_idx],
                        all_targets[idx],
                    )
                    break

    # 결정된 정답(all_targets)에 맞춰 스왑 실행
    final_choices = []
    final_answers = []

    for idx, row in df.iterrows():
        current_choices = list(row["choices"])
        target_ans = all_targets[idx]

        current_ans_idx = int(row["answer"]) - 1
        target_ans_idx = target_ans - 1

        # 실제 리스트 내 텍스트 위치 교체
        current_choices[current_ans_idx], current_choices[target_ans_idx] = (
            current_choices[target_ans_idx],
            current_choices[current_ans_idx],
        )

        final_choices.append(current_choices)
        final_answers.append(target_ans)

    df["choices"] = final_choices
    df["answer"] = final_answers

    return df


# -----------------------------
# 체크포인트 경로
# -----------------------------
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

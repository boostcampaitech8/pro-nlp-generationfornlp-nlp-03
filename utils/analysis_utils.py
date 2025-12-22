import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def analyze_subject_accuracy(
    df,
    true_col="label",     # 진짜 정답
    pred_col="answer",    # 모델 예측
    topic_col="topic",    # 과목 (사회, 경제, …)
    save_dir=None
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
        .agg(
            total_count=("correct", "size"),
            correct_count=("correct", "sum")
        )
        .reset_index()
    )

    result_df["correct_ratio"] = (
        result_df["correct_count"] / result_df["total_count"]
    )

    # 정답률 낮은 과목부터 정렬
    result_df = result_df.sort_values("correct_ratio")

    # -----------------------------
    # 표 출력 (보고용)
    # -----------------------------
    print(f"\n📊 {topic_col}-wise Accuracy Report")
    print(
        result_df.rename(columns={
            topic_col: "Subject",
            "total_count": "Total",
            "correct_count": "Correct",
            "correct_ratio": "Accuracy"
        })
    )

    # -----------------------------
    # 시각화 (보고용)
    # -----------------------------
    plt.figure(figsize=(11, max(4, len(result_df) * 0.38)))

    colors = [
        "firebrick" if r < 0.3 else
        "darkorange" if r < 0.6 else
        "seagreen"
        for r in result_df["correct_ratio"]
    ]

    bars = plt.barh(
        result_df[topic_col],
        result_df["correct_ratio"],
        color=colors
    )

    for _, row in result_df.iterrows():
        plt.text(
            row["correct_ratio"] + 0.01,
            row[topic_col],
            f"{row['correct_count']} / {row['total_count']}  ({row['correct_ratio']:.2f})",
            va="center",
            fontsize=10
        )

    plt.xlim(0, 1)
    plt.xlabel("Accuracy (Correct / Total)")
    plt.title(
        f"{topic_col}-wise Accuracy\n(How many questions were answered correctly per {topic_col})",
        fontsize=14,
        weight="bold"
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/{topic_col}_accuracy.png")

    plt.show()

    return result_df





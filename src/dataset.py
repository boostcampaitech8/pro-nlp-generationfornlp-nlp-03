from typing import List
import pandas as pd

from tqdm import tqdm
from ast import Dict, literal_eval
from datasets import Dataset

from src.utils import parse_choices

from src.prompt_template import (
    PROMPT_NO_QUESTION_PLUS,
    PROMPT_QUESTION_PLUS,
    SYSTEM_MESSAGE,
)


def load_data(path, mode="train"):

    print(path)
    dataset = pd.read_csv(path)

    records = []

    # 원본 데이터 처리
    if "question" not in dataset.columns:
        for _, row in dataset.iterrows():
            problems = literal_eval(row["problems"])
            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": problems["question"],
                "choices": problems["choices"],
                "answer": problems.get("answer", None),
                "question_plus": problems.get("question_plus", None),
                "choice_len": len(problems["choices"]),
                "topic": row.get("topic", None),
            }
            records.append(record)

    # 데이터 증강된 데이터 처리
    else:
        for _, row in dataset.iterrows():
            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": row["question"],
                "choices": parse_choices(row["choices"]),
                "answer": row.get("answer", None),
                "question_plus": row.get("question_plus", None),
                "topic": row.get("topic", None),
                "type": row["type"],
                "choice_len": len(parse_choices(row["choices"])),
                "stratify_key": row["stratify_key"],
            }
            records.append(record)

    return pd.DataFrame(records)


class MyDataset:
    """
    A dataset processing class that prepares data for training and testing based on provided configurations.

    Args:
        cfg (dict): Configuration dictionary containing prompt name and uniform answer distribution flag.
    """

    def __init__(self, config):
        self.uniform_answer_distribution = config["uniform_answer_distribution"]

    def create_user_message(self, row):

        def format_choices(choices: List[str]) -> str:
            """선택지를 문자열로 포맷팅"""
            return "\n".join(
                [f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)]
            )

        """데이터 행에서 user 메시지 생성"""
        choices_string = format_choices(row["choices"])

        if row["question_plus"]:
            return PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choice_count=row["choice_len"],
                choices=choices_string,
            )
        else:
            return PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choice_count=row["choice_len"],
                choices=choices_string,
            )

    def process_dataset(self, df, mode="train"):
        """
        학습용 데이터셋 전처리

        Args:
            df: 원본 DataFrame

        Returns:
            전처리된 HuggingFace Dataset
        """

        processed_data = []

        if mode == "train":
            for _, row in df.iterrows():
                user_message = self.create_user_message(row)

                processed_data.append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": str(row["answer"])},
                        ],
                        "label": row["answer"],
                    }
                )

            return Dataset.from_pandas(pd.DataFrame(processed_data))

        elif mode == "test":
            for _, row in df.iterrows():
                user_message = self.create_user_message(row)

                processed_data.append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": user_message},
                        ],
                        "label": row.get("answer"),  # test 데이터에는 없을 수 있음
                        "topic": row.get("topic"),
                        "type": row.get("type"),
                        "stratify_key": row.get("stratify_key"),
                        "len_choices": row["choice_len"],
                    }
                )

            return processed_data

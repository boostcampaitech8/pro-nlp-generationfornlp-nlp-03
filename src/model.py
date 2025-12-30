import unsloth
import os
import torch
import evaluate
import numpy as np
import pandas as pd
import random

from datasets import Dataset
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)

from src.utils import (
    analyze_subject_accuracy,
    create_metric_functions,
    get_torch_dtype,
    get_token_statistics,
)

from sklearn.metrics import f1_score, accuracy_score

# =============================================================================
# 토큰화 함수
# =============================================================================


# Not Instruct용 Chat Template
CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\\n' }}{% endif %}{% endfor %}"


def setup_tokenizer(tokenizer):
    """토크나이저 설정"""
    # Chat template 설정 (없는 경우)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE

    # Pad token 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return tokenizer


def tokenize_dataset(
    dataset: Dataset, tokenizer, max_seq_length: int = 1024, num_proc: int = 4
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
        lambda x: len(x["input_ids"]) <= max_seq_length, desc="Filtering by length"
    )

    return tokenized_dataset


# =============================================================================
# 모델 훈련
# =============================================================================


class MyModel:
    """
    Handle training and inference for a llm model with unsloth

    Args:
        config (dict): : Configuration dictionary for model and training setup.
        mode (str): Operating mode ("train", "inference")
    """

    def __init__(self, config, mode):
        self.config = config
        self.model_c = config["model"]
        self.visual_c = config["visualization"]
        self.peft_c = config["peft"]
        self.unsloth_c = config["UnslothTrainingArguments"]
        self.exp_name = self.config["model"]["experiment_name"]

        if mode == "train":
            model_name = self.model_c["train"]["model_name"]
        else:
            model_name = self.model_c["test"]["model_name"].format(
                experiment_name=self.exp_name
            )

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            dtype=get_torch_dtype(config["torch_dtype"]),
            load_in_4bit=True,
            max_seq_length=self.config["max_seq_length"],
        )

        self.tokenizer = setup_tokenizer(self.tokenizer)

        if mode == "train":
            self.model = FastLanguageModel.get_peft_model(
                model=self.model,
                r=self.peft_c["r"],
                target_modules=self.peft_c["target_modules"],
                lora_alpha=self.peft_c["lora_alpha"],
                lora_dropout=self.peft_c["lora_dropout"],
                bias=self.peft_c["bias"],
                use_gradient_checkpointing=self.peft_c["use_gradient_checkpointing"],
                random_state=config["seed"],
                use_rslora=self.peft_c["use_rslora"],
                loftq_config=None,
            )

        elif mode == "test":
            self.model = FastLanguageModel.for_inference(self.model)

    def train(self, processed_data):

        checkpoint_dir = self.model_c["train"]["train_checkpoint_path"].format(
            experiment_name=self.exp_name
        )

        print(f"\n📝 토큰화 중 (max_seq_length: {self.config['max_seq_length']})...")

        self.tokenized_dataset = tokenize_dataset(
            processed_data, self.tokenizer, max_seq_length=self.config["max_seq_length"]
        )

        if self.config["train_valid_split"]:
            split_dataset = self.tokenized_dataset.train_test_split(
                test_size=0.1, seed=self.config["seed"]
            )
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]

            print(f"  - Train: {len(self.train_dataset)} samples")
            print(f"  - Eval: {len(self.eval_dataset)} samples")

        else:
            self.train_dataset = self.tokenized_dataset
            self.eval_dataset = None

            print(f"  - Train: {len(self.train_dataset)} samples")

        stats = get_token_statistics(self.train_dataset, self.tokenizer)
        print(
            f"  - Train Token 길이: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}"
        )

        print(f"\n⚙️ LoRA 설정")
        print(f"  - r: {self.peft_c['r']}, alpha: {self.peft_c['lora_alpha']}")
        print(f"  - target_modules: {self.peft_c['target_modules']}")

        preprocess_logits_for_metrics, compute_metrics = create_metric_functions(
            self.tokenizer
        )

        print(f"\n📋 학습 설정")
        print(f"  - epochs: {self.unsloth_c['num_train_epochs']}")
        print(f"  - batch_size: {self.unsloth_c['per_device_train_batch_size']}")
        print(f"  - learning_rate: {self.unsloth_c['learning_rate']}")
        print(f"  - output_dir: {self.model_c['train']['train_checkpoint_path']}")

        eval_strategy = self.unsloth_c["eval_strategy"]

        if self.eval_dataset is None:
            eval_strategy = "no"

        args = UnslothTrainingArguments(
            do_train=self.unsloth_c["do_train"],
            do_eval=self.unsloth_c["do_eval"],
            per_device_train_batch_size=self.unsloth_c["per_device_train_batch_size"],
            per_device_eval_batch_size=self.unsloth_c["per_device_eval_batch_size"],
            num_train_epochs=self.unsloth_c["num_train_epochs"],
            gradient_accumulation_steps=self.unsloth_c["gradient_accumulation_steps"],
            learning_rate=float(self.unsloth_c["learning_rate"]),
            embedding_learning_rate=float(self.unsloth_c["embedding_learning_rate"]),
            fp16=not is_bfloat16_supported(),  # Use FP16 if BF16 is not supported
            bf16=is_bfloat16_supported(),
            weight_decay=self.unsloth_c["weight_decay"],
            lr_scheduler_type=self.unsloth_c["lr_scheduler_type"],
            warmup_ratio=self.unsloth_c["warmup_ratio"],
            warmup_steps=self.unsloth_c["warmup_steps"],
            optim=self.unsloth_c["optim"],
            logging_steps=self.unsloth_c["logging_steps"],
            save_strategy=self.unsloth_c["save_strategy"],
            eval_strategy=eval_strategy,
            save_total_limit=self.unsloth_c["save_total_limit"],
            save_only_model=self.unsloth_c["save_only_model"],
            # report_to=self.unsloth_c["report_to"]
            output_dir=checkpoint_dir,
        )

        trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=args,
        )

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"📂 폴더 생성 완료: {checkpoint_dir}")

        # 9. 학습 실행
        print("\n" + "=" * 60)
        print("🏃 학습 실행 중...")
        print("=" * 60)

        train_result = trainer.train()

        # 10. 결과 출력
        print("\n" + "=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)
        print(f"  - Total steps: {train_result.global_step}")
        print(f"  - Train loss: {train_result.training_loss:.4f}")
        print(f"  - 체크포인트 저장 위치: {checkpoint_dir}")

    def inference(self, processed_data):

        self.model_c["test"]["test_output_csv"] = self.model_c["test"][
            "test_output_csv"
        ].format(experiment_name=self.exp_name)

        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

        print("=" * 60)
        print("🔮 추론 시작")
        print("=" * 60)

        FastLanguageModel.for_inference(self.model)

        print("  - 모델 및 토크나이저 로드 완료")

        infer_results = []

        # validation 데이터 인 경우에 사용 - 새로 추가
        topics = []
        labels = []
        types = []
        types_topics = []

        self.model.eval()
        with torch.inference_mode():
            for data in tqdm(processed_data, desc="Inference"):
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                if data["topic"] is not None and data["label"] is not None:
                    topics.append(data["topic"])
                    labels.append(str(data["label"]))
                    types.append(data["type"])
                    types_topics.append(data["stratify_key"])

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")

                outputs = self.model(inputs)
                logits = outputs.logits[:, -1].flatten().cpu()

                # 선택지 토큰의 logit 추출
                target_logit_list = [
                    logits[self.tokenizer.vocab[str(i + 1)]] for i in range(len_choices)
                ]

                # Softmax로 확률 변환
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(target_logit_list, dtype=torch.float32), dim=-1
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                predict_value = pred_choices_map[np.argmax(probs)]
                # logit_confidence = float(np.max(probs))

                infer_results.append(
                    {
                        "id": _id,
                        "answer": predict_value,
                    }
                )

        output_dir = self.model_c["test"]["test_output_csv"]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"📂 폴더 생성 완료: {output_dir}")

        print(f"\n💾 결과 저장 중: {output_dir}/{self.exp_name}.csv")

        result_df = pd.DataFrame(infer_results)
        result_df.to_csv(
            f"{output_dir}/{self.exp_name}.csv",
            index=False,
            columns=["id", "answer"],
        )

        print("\n" + "=" * 60)
        print("✅ 추론 완료!")
        print("=" * 60)
        print(f"  - 총 예측 수: {len(infer_results)}")
        print(f"  - 저장 위치: {output_dir}/{self.exp_name}.csv")

        # 예측 분포 출력
        print("\n📊 예측 분포:")
        value_counts = result_df["answer"].value_counts().sort_index()
        for ans, count in value_counts.items():
            print(f"  - {ans}: {count} ({count/len(result_df)*100:.1f}%)")

        if len(topics) != 0:
            df = result_df.copy()

            df["label"] = labels
            df["topic"] = topics
            df["type"] = types
            df["stratify_key"] = types_topics

            f1_macro = f1_score(df["label"], df["answer"], average="macro")
            f1_weighted = f1_score(df["label"], df["answer"], average="weighted")
            acc = accuracy_score(df["label"], df["answer"])

            print("\n📑 Score Report:")
            print(f"Accuracy     : {acc:.4f}")
            print(f"F1-macro     : {f1_macro:.4f}")
            print(f"F1-weighted  : {f1_weighted:.4f}")

            if self.visual_c["choose_visualize"]:

                visual_path = self.visual_c["visualize_path"].format(
                    image_name=self.visual_c["image_name"]
                )

                result_topic = analyze_subject_accuracy(
                    df,
                    true_col="label",
                    pred_col="answer",
                    topic_col="topic",
                    save_dir=visual_path,
                )

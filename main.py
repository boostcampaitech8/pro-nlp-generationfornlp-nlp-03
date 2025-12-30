import unsloth
import os
import yaml
import argparse
import shutil
import pandas as pd

from src.dataset import MyDataset, load_data
from src.model import MyModel
from src.utils import balance_answer_by_swap, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="./config/config.yaml")
    parser.add_argument("--mode", "-m", type=str, default="train")
    parser.add_argument("--valid", "-v", type=bool, default=False)
    args = parser.parse_args()

    # Load YAML configuration file
    with open(args.config) as f:
        config = yaml.full_load(f)

    set_seed(config["seed"])
    dataset = MyDataset(config)
    model = MyModel(config, args.mode)

    if args.mode == "train":

        # load train data
        print("\n📂 데이터 로드 중...")
        train_df = load_data(
            path=config["model"]["train"]["train_data"], mode=args.mode
        )
        print(f"  - 총 데이터 수: {len(train_df)}")

        if config["uniform_answer_distribution"]:
            train_df = balance_answer_by_swap(train_df)
            print(f"\n📊선택지 비율: \n{train_df['answer'].value_counts()}\n")

        processed_df = dataset.process_dataset(train_df, args.mode)
        print(f"  - 전처리 완료: {len(processed_df)} samples")

        print(f"\n🤖 모델 로드 중: {config['model']['train']['model_name']}")
        model.train(processed_df)

    elif args.mode == "test":

        # load test data
        if args.valid:
            print("\n📂 검증 데이터 로드 중...")
            test_df = load_data(
                path=config["model"]["test"]["valid_data"], mode=args.mode
            )
        else:
            print("\n📂 테스트 데이터 로드 중...")
            test_df = load_data(
                path=config["model"]["test"]["test_data"], mode=args.mode
            )

        print(f"  - 총 데이터 수: {len(test_df)}")

        processed_df = dataset.process_dataset(test_df, args.mode)
        print(f"  - 전처리 완료: {len(processed_df)} samples")

        print(f"\n🤖 모델 로드 중: {config['model']['test']['model_name']}")

        model.inference(processed_df)

"""
실험 설정 파일
하이퍼파라미터와 경로를 여기서 수정하세요.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    """경로 설정"""
    train_data: str = "/data/ephemeral/home/data/train_data_final.csv"
    test_data: str = "/data/ephemeral/home/data/val_data_final.csv"   # 검증 데이터
    # test_data: str = "/data/ephemeral/home/data/test.csv"
    output_dir: str = "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-03/outputs_unsloth"
    output_csv: str = "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-03/output.csv"
    visualize_dir: str = "/data/ephemeral/home/images"


@dataclass
class ModelConfig:
    """모델 설정"""
    model_name: str = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA 설정"""
    r: int = 8                              # LoRA rank (높을수록 표현력↑, 메모리↑) / 나중에 튜닝할 것
    lora_alpha: int = 16                     # 스케일링 파라미터 / 나중에 튜닝할 것
    lora_dropout: float = 0.05              # 드롭아웃
    target_modules: List[str] = field(      # 학습할 모듈
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj']
        # 확장 가능: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """학습 설정"""
    # 기본 설정
    seed: int = 42
    max_seq_length: int = 4096         # 최대 시퀀스 길이 (힌트: 늘리면 성능↑)
    
    # 배치/에폭 설정
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 16    # 실질 배치 = batch_size * accumulation
    
    # 옵티마이저 설정
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"       # "linear", "cosine", "constant"
    warmup_ratio: float = 0.0
    warmup_steps: int = 5
    max_steps: int = 60
    optim: str = "adamw_8bit"
    
    # 로깅/저장 설정
    logging_steps: int = 1
    save_strategy: str = "epoch"            # "epoch", "steps"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 1
    save_only_model: bool = True
    
    # 평가 설정
    test_size: float = 0.0                # validation split 비율


@dataclass
class Config:
    """전체 설정"""
    path: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# 기본 설정 인스턴스
def get_config() -> Config:
    return Config()


# 실험별 설정 예시
def get_experiment_config(exp_name: str) -> Config:
    """
    실험별로 다른 설정을 반환
    
    사용법:
        config = get_experiment_config("large_context")
    """
    config = get_config()
    
    if exp_name == "large_context":
        # 더 긴 컨텍스트 실험
        config.training.max_seq_length = 4096
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 2
        
    elif exp_name == "more_lora":
        # 더 많은 LoRA 모듈 학습
        config.lora.r = 16
        config.lora.lora_alpha = 32
        config.lora.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
    elif exp_name == "longer_training":
        # 더 오래 학습
        config.training.num_train_epochs = 5
        config.training.learning_rate = 1e-5
        
    return config

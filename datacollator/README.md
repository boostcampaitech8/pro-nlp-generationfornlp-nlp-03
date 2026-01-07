# 🎯 Generation for NLP Baseline

한국어 객관식 문제 풀이 베이스라인 코드입니다.

## 📁 파일 구조

```
baseline/
├── config.py        # 설정 파일 (하이퍼파라미터, 경로)
├── data_utils.py    # 데이터 처리 유틸리티
├── train.py         # 학습 스크립트
├── inference.py     # 추론 스크립트
├── requirements.txt # 의존성 패키지
└── README.md        # 사용 가이드
```

## 🚀 Quick Start

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python3.10 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

`train.csv`와 `test.csv`를 현재 디렉토리에 위치시키세요.

### 3. 학습

```bash
# 기본 학습
python train.py

# 실험 설정으로 학습
python train.py --exp large_context

# 커스텀 설정
python train.py --epochs 5 --lr 1e-5 --max_seq_length 2048
```

### 4. 추론

```bash
# 기본 추론
python inference.py --checkpoint outputs/checkpoint-4491

# 커스텀 출력 경로
python inference.py --checkpoint outputs/checkpoint-4491 --output my_result.csv
```

---

## ⚙️ 설정 변경하기

### 방법 1: config.py 직접 수정

```python
# config.py 에서 직접 수정
@dataclass
class TrainingConfig:
    num_train_epochs: int = 5      # 3 → 5
    learning_rate: float = 1e-5    # 2e-5 → 1e-5
```

### 방법 2: 명령줄 인자 사용

```bash
python train.py --epochs 5 --lr 1e-5 --max_seq_length 2048 --output_dir my_exp
```

### 방법 3: 실험 프리셋 사용

```bash
# 사전 정의된 실험 설정 사용
python train.py --exp large_context   # 더 긴 컨텍스트
python train.py --exp more_lora       # 더 많은 LoRA 모듈
python train.py --exp longer_training # 더 긴 학습
```

---

## 🔬 실험 가이드

### 실험 1: 더 긴 시퀀스 처리

```python
# config.py 수정
config.training.max_seq_length = 2048  # 1024 → 2048
config.training.per_device_train_batch_size = 1
config.training.gradient_accumulation_steps = 2  # 메모리 절약
```

또는:
```bash
python train.py --exp large_context
```

### 실험 2: LoRA 확장

```python
# config.py 수정
config.lora.r = 16  # 6 → 16
config.lora.lora_alpha = 32  # 8 → 32
config.lora.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
```

또는:
```bash
python train.py --exp more_lora
```

### 실험 3: 다른 모델 사용

```python
# config.py 수정
config.model.model_name = "beomi/gemma-ko-7b"  # 더 큰 모델
```

### 실험 4: Learning Rate 스케줄 변경

```python
# config.py 수정
config.training.lr_scheduler_type = "linear"  # "cosine" → "linear"
config.training.warmup_ratio = 0.1  # warmup 추가
```

---

## 📊 성능 개선 힌트

1. **데이터**
   - `max_seq_length` 늘리기 (1024 초과 데이터 포함)
   - 데이터 증강 (선택지 순서 섞기)

2. **모델**
   - 더 큰 모델 사용 (gemma-ko-7b)
   - LoRA rank(r) 증가

3. **학습**
   - 에폭 수 증가
   - Learning rate 조정
   - Gradient accumulation 활용

4. **앙상블**
   - 여러 체크포인트 결과 조합

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# 배치 사이즈 줄이기
config.training.per_device_train_batch_size = 1

# Gradient accumulation 사용
config.training.gradient_accumulation_steps = 4

# 시퀀스 길이 줄이기
config.training.max_seq_length = 512
```

### 학습이 느릴 때

```python
# Mixed precision 사용 (이미 float16)
config.model.torch_dtype = "bfloat16"  # A100 등에서

# 로깅 빈도 줄이기
config.training.logging_steps = 50
```

---

## 📝 주요 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `max_seq_length` | 1024 | 최대 입력 토큰 수 |
| `num_train_epochs` | 3 | 학습 에폭 수 |
| `learning_rate` | 2e-5 | 학습률 |
| `lora.r` | 6 | LoRA rank (표현력) |
| `lora.lora_alpha` | 8 | LoRA 스케일링 |
| `lora.target_modules` | q_proj, k_proj | 학습 대상 모듈 |

---

## 📌 베이스라인 성능

- Validation Accuracy: ~47%
- 학습 시간: ~18분 (3 epochs)
- 추론 시간: ~15분

---

## License

This code is for educational purposes.

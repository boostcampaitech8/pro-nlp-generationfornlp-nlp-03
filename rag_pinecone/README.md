# RAG 실행 가이드

## 환경 설정

```bash
export UPSTAGE_API_KEY='your-upstage-api-key'
export PINECONE_API_KEY='your-pinecone-api-key'
```

## 실행 순서

### 1. RAG Facts 수집

```bash
python ./rag/prepare_rag_enhanced_data.py \
    --input ../data/valid_topic_sub-2.csv \
    --output ../data/rag/valid_with_rag.csv \
    --use-choice-search \
    --use-item-search
```

### 2. RAG-Enhanced Inference

```bash
python inference_second.py \
    --checkpoint outputs_gemma/checkpoint-4778 \
    --test_data ../data/rag/valid_with_rag.csv \
    --output valid_submission_with_rag.csv \
    --use_rag \
    --save_details
```

## 주요 파라미터

- `--use-choice-search`: 선택지별 검색
- `--use-item-search`: 보기별 검색 (가, 나, 다, 라 등)
- `--use_rag`: RAG facts를 프롬프트에 포함
- `--save_details`: 상세 정보 JSON 저장

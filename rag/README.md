# 한국사 3단계 Hybrid RAG 시스템

Dense Retrieval + BM25 + Reranker로 한국사 객관식 문제를 풉니다.

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 준비

필수 파일:

- `data/raw/rag_output.json` (한국사 원본 문서)
- `train_topic_sub-2.csv` (평가 문제)

## 실행 순서

### 1. 데이터 청킹

```bash
python chunker.py
```

→ `data/chunks/chunks.pkl` 생성

### 2. Dense 인덱스 구축

```bash
python build_dense.py \
    --chunks-path data/chunks/chunks.pkl \
    --index-path models/index_opensource
```

→ `models/index_opensource.index`, `models/index_opensource.mapping` 생성

### 3. BM25 인덱스 구축

```bash
python build_bm25.py \
    --chunk-file data/chunks/chunks.pkl \
    --output models/bm25_index.pkl
```

→ `models/bm25_index.pkl` 생성

### 4. 평가 실행

```bash
python evaluate_hybrid_rag.py \
    --data-path 평가할 데이터셋 \
    --output results.json
```

→ `results.json` 생성

## 옵션

```bash
# 상세 로그
python evaluate_hybrid_rag.py --data-path train_topic_sub-2.csv --verbose

# Reranker 없이
python evaluate_hybrid_rag.py --data-path train_topic_sub-2.csv --no-reranker
```

## 시스템 구조

```
Step 1: 지문 + 질문 검색 → "경덕왕" 추출
Step 2: "경덕왕 밑줄 친 '왕'의 재위 기간에..."
Step 3: 각 선지 검색 → 점수 계산
```

"""
Solar Embeddings API로 Dense Retrieval 인덱스 구축
Upstage Solar Embeddings 사용
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import time

try:
    from openai import OpenAI
except ImportError:
    logger.error("openai not installed. Run: pip install openai")
    sys.exit(1)

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval_dense import DenseRetrieval


def build_index_solar(
    chunks_path="data/chunks/chunks.pkl",
    index_path="models/index_solar",
    model_name="solar-embedding-1-large-query",
    batch_size=100,
    api_key=None
):
    """
    Solar Embeddings API로 인덱스 구축
    
    Args:
        chunks_path: 청크 파일 경로
        index_path: 인덱스 저장 경로
        model_name: Solar 임베딩 모델 이름
            - solar-embedding-1-large-query (쿼리용, 4096차원)
            - solar-embedding-1-large-passage (문서용, 4096차원)
        batch_size: 배치 크기
        api_key: Upstage API 키
    """
    logger.info("=" * 80)
    logger.info("Building Dense Retrieval Index with Solar Embeddings")
    logger.info("=" * 80)
    
    # API 키 설정
    if api_key:
        os.environ["UPSTAGE_API_KEY"] = api_key
    
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY not found!")
        logger.error("Set: export UPSTAGE_API_KEY='your-key'")
        sys.exit(1)
    
    # Solar API 클라이언트 (OpenAI 호환)
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1/solar"
    )
    
    # 청크 로드
    logger.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # 텍스트 추출
    texts = [chunk['text'] for chunk in chunks]
    metadata = chunks
    
    # 임베딩 생성
    logger.info(f"Generating embeddings with {model_name}...")
    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}...")
    
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", total=total_batches):
        batch_texts = texts[i:i + batch_size]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model=model_name
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # 성공 시 정상 대기
                time.sleep(2)  # 2초 대기
                break
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30, 60초
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error at batch {i//batch_size}: {e}")
                    raise
    
    embeddings = np.array(embeddings)
    embedding_dim = embeddings.shape[1]
    
    logger.info(f"Generated embeddings: {embeddings.shape}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # 정규화
    logger.info("Normalizing embeddings...")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 인덱스 구축
    logger.info("Building FAISS index...")
    retriever = DenseRetrieval(
        embedding_dim=embedding_dim,
        index_type="flat"
    )
    retriever.build_index(embeddings, texts, metadata)
    
    # 저장
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    retriever.save(index_path)
    
    logger.info("=" * 80)
    logger.info(f"Index saved to {index_path}")
    logger.info(f"Total documents: {len(texts)}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Model used: {model_name}")
    logger.info("=" * 80)
    
    return retriever


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Dense Index with Solar Embeddings")
    parser.add_argument(
        '--chunks-path',
        type=str,
        default='data/chunks/chunks.pkl',
        help='Path to chunks file'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='models/index_solar',
        help='Path to save index'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='solar-embedding-1-large-passage',
        help='Solar embedding model (passage for documents, query for queries)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Upstage API key (or set UPSTAGE_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    build_index_solar(
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        api_key=args.api_key
    )
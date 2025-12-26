"""
오픈소스 임베딩 모델로 Dense Retrieval 인덱스 구축
sentence-transformers 사용
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

# 상위 디렉토리를 Python 경로에 추가
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval_dense import DenseRetrieval

def build_index_opensource(
    chunks_path="data/processed/chunks.pkl",
    index_path="models/index_opensource",
    model_name="jhgan/ko-sroberta-multitask",
    batch_size=32
):
    """
    오픈소스 모델로 인덱스 구축
    
    Args:
        chunks_path: 청크 파일 경로
        index_path: 인덱스 저장 경로  
        model_name: 임베딩 모델 이름
        batch_size: 배치 크기
    """
    logger.info("=" * 80)
    logger.info("Building Dense Retrieval Index with Open-source Model")
    logger.info("=" * 80)
    
    # 청크 로드
    logger.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # 임베딩 모델 로드
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded. Embedding dimension: {embedding_dim}")
    
    # 텍스트 추출
    texts = [chunk['text'] for chunk in chunks]
    metadata = chunks
    
    # 임베딩 생성
    logger.info("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    logger.info(f"Generated embeddings: {embeddings.shape}")
    
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
    logger.info("=" * 80)
    
    return retriever


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Dense Retrieval Index")
    parser.add_argument(
        '--chunks-path',
        type=str,
        default='data/processed/chunks.pkl',
        help='Path to chunks file'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='models/index_opensource',
        help='Path to save index'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='jhgan/ko-sroberta-multitask',
        help='Sentence-transformers model name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )
    
    args = parser.parse_args()
    
    build_index_opensource(
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
"""
retrieval_bm25.py
BM25 Sparse Retrieval 모듈
키워드 기반 검색 기능 제공
"""
import json
import os
import pickle
from typing import Dict, List, Optional
from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retrieval:
    """BM25 기반 Sparse Retrieval 클래스"""
    
    def __init__(
        self,
        tokenize_fn=None,
        index_path: Optional[str] = None
    ):
        """
        Args:
            tokenize_fn: 토크나이제이션 함수
            index_path: 기존 인덱스 로드 경로
        """
        self.tokenize_fn = tokenize_fn if tokenize_fn else self._default_tokenizer
        self.bm25 = None
        self.corpus = []
        self.metadata = []
        
        # 기존 인덱스 로드
        if index_path and os.path.exists(index_path):
            self.load(index_path)
            
    def _default_tokenizer(self, text: str) -> List[str]:
        """기본 토크나이저 (공백 기반)"""
        return text.split()
    
    def build_index(
        self,
        documents: List[str],
        metadata: List[Dict] = None
    ):
        """
        BM25 인덱스 구축
        
        Args:
            documents: 문서 텍스트 리스트
            metadata: 메타데이터 리스트
        """
        self.corpus = documents
        self.metadata = metadata if metadata else [{} for _ in documents]
        
        # 문서 토크나이제이션
        logger.info("Tokenizing corpus...")
        tokenized_corpus = [self.tokenize_fn(doc) for doc in documents]
        
        # BM25 인덱스 생성
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"Built BM25 index with {len(documents)} documents")
        
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        쿼리에 대한 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            검색 결과 리스트
        """
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_index() first.")
            
        # 쿼리 토크나이제이션
        tokenized_query = self.tokenize_fn(query)
        
        # BM25 스코어 계산
        scores = self.bm25.get_scores(tokenized_query)
        
        # 상위 k개 문서 인덱스
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # 결과 구성
        results = []
        for idx in top_indices:
            result = {
                "text": self.corpus[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            }
            results.append(result)
            
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        여러 쿼리에 대한 배치 검색
        
        Args:
            queries: 검색 쿼리 리스트
            top_k: 반환할 문서 수
            
        Returns:
            각 쿼리에 대한 검색 결과 리스트
        """
        results = []
        for query in queries:
            query_results = self.retrieve(query, top_k)
            results.append(query_results)
            
        return results
    
    def save(self, index_path: str):
        """인덱스 저장"""
        data = {
            "bm25": self.bm25,
            "corpus": self.corpus,
            "metadata": self.metadata
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved BM25 index to {index_path}")
        
    def load(self, index_path: str):
        """인덱스 로드"""
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            
        self.bm25 = data["bm25"]
        self.corpus = data["corpus"]
        self.metadata = data["metadata"]
        
        logger.info(f"Loaded BM25 index from {index_path}")
        logger.info(f"Index contains {len(self.corpus)} documents")


if __name__ == "__main__":
    # 예제 실행
    documents = [
        "조선 시대는 1392년부터 1910년까지 존속한 한국의 왕조입니다.",
        "고려 시대는 918년부터 1392년까지 한반도를 통치했습니다.",
        "삼국 시대는 고구려, 백제, 신라가 경쟁한 시기입니다."
    ]
    
    # 인덱스 구축
    retriever = BM25Retrieval()
    retriever.build_index(documents)
    
    # 검색
    query = "조선 왕조의 역사"
    results = retriever.retrieve(query, top_k=2)
    
    print("Search Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text']}")
        print(f"   Score: {result['score']:.4f}\n")
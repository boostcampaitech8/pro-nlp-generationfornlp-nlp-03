"""
reranker.py
Reranking 모듈
검색 결과를 재정렬하는 기능 제공
"""
import gc
from typing import Dict, List
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    """Cross-encoder 기반 Reranker 클래스"""
    
    def __init__(
        self,
        model_path: str = "Dongjin-kr/ko-reranker",
        batch_size: int = 128,
        max_length: int = 512,
        device: str = "cuda:0"
    ):
        """
        Args:
            model_path: Reranking 모델 경로
            batch_size: 배치 크기
            max_length: 최대 시퀀스 길이
            device: 사용할 디바이스
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 모델 및 토크나이저 로드
        logger.info(f"Loading reranker model: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Reranker model loaded successfully")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """메모리 정리"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    def _exp_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Exponential normalization"""
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        단일 쿼리에 대한 문서 재정렬
        
        Args:
            query: 검색 쿼리
            documents: 재정렬할 문서 리스트
            top_k: 반환할 상위 문서 수
            
        Returns:
            재정렬된 문서 리스트
        """
        # 쿼리-문서 쌍 생성
        pairs = [[query, doc["text"]] for doc in documents]
        
        # 배치 처리
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            with torch.no_grad():
                # 토크나이제이션
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 모델 추론
                outputs = self.model(**inputs, return_dict=True)
                batch_scores = outputs.logits.view(-1).float().cpu().numpy()
                all_scores.extend(batch_scores)
        
        # 스코어 정규화
        scores = np.array(all_scores)
        scores = self._exp_normalize(scores.reshape(1, -1)).flatten()
        
        # 상위 k개 인덱스
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # 결과 구성
        reranked_results = []
        for idx in top_indices:
            result = {
                "text": documents[idx]["text"],
                "score": float(scores[idx]),
                "metadata": documents[idx].get("metadata", {})
            }
            reranked_results.append(result)
            
        return reranked_results
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict]],
        top_k: int = 10,
        show_progress: bool = True
    ) -> List[List[Dict]]:
        """
        여러 쿼리에 대한 배치 재정렬
        
        Args:
            queries: 검색 쿼리 리스트
            documents_list: 각 쿼리에 대한 문서 리스트
            top_k: 반환할 상위 문서 수
            show_progress: 진행률 표시 여부
            
        Returns:
            각 쿼리에 대한 재정렬된 문서 리스트
        """
        # 모든 쿼리-문서 쌍 생성
        all_pairs = []
        pair_boundaries = [0]
        
        for query, documents in zip(queries, documents_list):
            for doc in documents:
                all_pairs.append([query, doc["text"]])
            pair_boundaries.append(len(all_pairs))
        
        # 배치 처리
        all_scores = []
        iterator = tqdm(
            range(0, len(all_pairs), self.batch_size),
            desc="Reranking"
        ) if show_progress else range(0, len(all_pairs), self.batch_size)
        
        for i in iterator:
            batch_pairs = all_pairs[i:i + self.batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, return_dict=True)
                batch_scores = outputs.logits.view(-1).float().cpu().numpy()
                all_scores.extend(batch_scores)
        
        all_scores = np.array(all_scores)
        
        # 각 쿼리별로 결과 분리
        reranked_results = []
        for i in range(len(queries)):
            start_idx = pair_boundaries[i]
            end_idx = pair_boundaries[i + 1]
            
            scores = all_scores[start_idx:end_idx]
            scores = self._exp_normalize(scores.reshape(1, -1)).flatten()
            
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            query_results = []
            for idx in top_indices:
                result = {
                    "text": documents_list[i][idx]["text"],
                    "score": float(scores[idx]),
                    "metadata": documents_list[i][idx].get("metadata", {})
                }
                query_results.append(result)
                
            reranked_results.append(query_results)
            
        return reranked_results


if __name__ == "__main__":
    # 예제 실행
    query = "조선 시대의 역사"
    documents = [
        {"text": "조선 시대는 1392년부터 1910년까지 존속했습니다.", "metadata": {}},
        {"text": "고려 시대는 918년부터 1392년까지 존속했습니다.", "metadata": {}},
        {"text": "조선왕조는 태조 이성계가 건국했습니다.", "metadata": {}}
    ]
    
    # Reranker 초기화
    with Reranker(model_path="Dongjin-kr/ko-reranker") as reranker:
        # 재정렬
        results = reranker.rerank(query, documents, top_k=2)
        
        print("Reranked Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text']}")
            print(f"   Score: {result['score']:.4f}\n")
"""
retrieval_dense.py
Dense Retrieval 모듈
FAISS 기반 벡터 검색 기능 제공
"""
import pickle
from typing import Dict, List, Tuple
import numpy as np
import torch
from loguru import logger
import faiss


class DenseRetrieval:
    """FAISS 기반 Dense Retrieval 클래스"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "flat",
        device: str = "cuda:0"
    ):
        """
        Args:
            embedding_dim: 임베딩 차원
            index_type: FAISS 인덱스 타입 (flat, hnsw)
            device: 사용할 디바이스
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # FAISS 인덱스 초기화
        self.index = self._initialize_index()
        self.id_to_text = {}  # 인덱스 ID -> 텍스트 매핑
        
    def _initialize_index(self):
        """FAISS 인덱스 초기화"""
        if self.index_type == "flat":
            # Inner Product 기반 Flat 인덱스
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) 인덱스
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
            
        logger.info(f"Initialized {self.index_type} FAISS index with dim={self.embedding_dim}")
        return index
    
    def build_index(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: List[Dict] = None
    ):
        """
        임베딩으로 인덱스 구축
        
        Args:
            embeddings: 문서 임베딩 (N x D)
            texts: 문서 텍스트 리스트
            metadata: 메타데이터 리스트
        """
        # 임베딩 정규화 (cosine similarity를 위해)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # FAISS 인덱스에 추가
        self.index.add(embeddings.astype('float32'))
        
        # ID to text 매핑 저장
        for idx, text in enumerate(texts):
            self.id_to_text[idx] = {
                "text": text,
                "metadata": metadata[idx] if metadata else {}
            }
        
        logger.info(f"Built index with {len(texts)} documents")
        
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """
        쿼리에 대한 유사 문서 검색
        
        Args:
            query_embedding: 쿼리 임베딩 (1 x D)
            top_k: 반환할 문서 수
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩 정규화
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # FAISS 검색
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS에서 찾지 못한 경우
                continue
                
            result = {
                "text": self.id_to_text[idx]["text"],
                "score": float(score),
                "metadata": self.id_to_text[idx]["metadata"]
            }
            results.append(result)
            
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        배치 검색
        
        Args:
            query_embeddings: 쿼리 임베딩 배치 (N x D)
            top_k: 반환할 문서 수
            
        Returns:
            각 쿼리에 대한 검색 결과 리스트
        """
        # 쿼리 임베딩 정규화
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        query_embeddings = query_embeddings.astype('float32')
        
        # FAISS 배치 검색
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # 결과 구성
        all_results = []
        for batch_scores, batch_indices in zip(scores, indices):
            results = []
            for score, idx in zip(batch_scores, batch_indices):
                if idx == -1:
                    continue
                    
                result = {
                    "text": self.id_to_text[idx]["text"],
                    "score": float(score),
                    "metadata": self.id_to_text[idx]["metadata"]
                }
                results.append(result)
            all_results.append(results)
            
        return all_results
    
    def save(self, index_path: str):
        """인덱스와 매핑 저장"""
        # FAISS 인덱스 저장
        faiss.write_index(self.index, f"{index_path}.index")
        
        # ID to text 매핑 저장
        with open(f"{index_path}.mapping", 'wb') as f:
            pickle.dump(self.id_to_text, f)
            
        logger.info(f"Saved index to {index_path}")
        
    def load(self, index_path: str):
        """인덱스와 매핑 로드"""
        # FAISS 인덱스 로드
        self.index = faiss.read_index(f"{index_path}.index")
        
        # ID to text 매핑 로드
        with open(f"{index_path}.mapping", 'rb') as f:
            self.id_to_text = pickle.load(f)
            
        logger.info(f"Loaded index from {index_path}")
        logger.info(f"Index contains {self.index.ntotal} vectors")


class EmbeddingGenerator:
    """임베딩 생성 클래스"""
    
    def __init__(
        self,
        model_name: str = "SOLAR-10.7B-Instruct-v1.0",
        api_key: str = None,
        batch_size: int = 32,
        device: str = "cuda:0"
    ):
        """
        Args:
            model_name: 사용할 모델 이름
            api_key: API 키 (SOLAR API 사용 시)
            batch_size: 배치 크기
            device: 사용할 디바이스
        """
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.device = device
        
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        텍스트 리스트에 대한 임베딩 생성
        
        Args:
            texts: 텍스트 리스트
            show_progress: 진행률 표시 여부
            
        Returns:
            임베딩 배열 (N x D)
        """
        # TODO: SOLAR API 또는 다른 임베딩 모델 사용
        # 여기서는 예시로 랜덤 임베딩 반환
        logger.warning("Using random embeddings for demonstration")
        
        embeddings = []
        from tqdm import tqdm
        
        iterator = tqdm(range(0, len(texts), self.batch_size)) if show_progress else range(0, len(texts), self.batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            # 실제로는 여기서 SOLAR API 또는 임베딩 모델 호출
            batch_embeddings = np.random.randn(len(batch_texts), 1024)
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)


if __name__ == "__main__":
    # 예제 실행
    texts = [
        "조선 시대의 역사",
        "고려 시대의 문화",
        "삼국 시대의 전쟁"
    ]
    
    # 임베딩 생성
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(texts)
    
    # 인덱스 구축
    retrieval = DenseRetrieval(embedding_dim=1024)
    retrieval.build_index(embeddings, texts)
    
    # 검색
    query_embedding = np.random.randn(1024)
    results = retrieval.search(query_embedding, top_k=2)
    
    print("Search Results:")
    for result in results:
        print(f"- {result['text']} (score: {result['score']:.4f})")
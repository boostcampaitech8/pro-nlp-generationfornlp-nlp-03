"""
Pinecone 검색 (섹션 메타데이터 매핑)
"""
import os
import json
from typing import List, Dict
from loguru import logger

try:
    from pinecone import Pinecone
except ImportError:
    logger.error("pinecone not installed")
    raise

try:
    from openai import OpenAI
except ImportError:
    logger.error("openai not installed")
    raise

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class PineconeDenseRetrievalWithJSON:
    """Pinecone + JSON 메타데이터 검색"""
    
    def __init__(
        self,
        index_name: str = "korean-history",
        sections_file: str = "../data/rag_output_sections.json",
        query_model: str = "solar-embedding-1-large-query"
    ):
        self.index_name = index_name
        self.query_model = query_model
        
        # API 키
        solar_api_key = os.getenv("UPSTAGE_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not solar_api_key or not pinecone_api_key:
            raise ValueError("API keys required")
        
        # Solar API
        self.solar_client = OpenAI(
            api_key=solar_api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
        
        # Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
        
        # 섹션 메타데이터 로드 (한글 포함!)
        logger.info(f"Loading sections metadata from {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            self.sections = json.load(f)
        
        logger.info(f"✅ Loaded {len(self.sections)} sections")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """쿼리 임베딩"""
        response = self.solar_client.embeddings.create(
            input=[query],
            model=self.query_model
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """검색"""
        # 쿼리 임베딩
        query_embedding = self._get_query_embedding(query)
        
        # Pinecone 검색
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=""
        )
        
        # 결과 구성 (섹션 메타데이터 매핑)
        output = []
        for match in results.matches:
            section_idx = match.metadata.get("section_idx", 0)
            
            # 섹션 정보 가져오기
            if section_idx < len(self.sections):
                section = self.sections[section_idx]
                
                output.append({
                    "text": section['text'],  # 한글 텍스트!
                    "score": float(match.score),
                    "metadata": {
                        "main_title": section['title'],
                        "sub_title": section['sub_title'],
                        "section_title": section['section_title'],
                        "doc_idx": section['doc_idx'],
                        "section_idx": section_idx
                    }
                })
        
        return output
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict]]:
        """배치 검색"""
        results = []
        for query in queries:
            results.append(self.search(query, top_k))
        return results


if __name__ == "__main__":
    # 테스트
    retriever = PineconeDenseRetrievalWithJSON(
        index_name="korean-history",
        sections_file="../data/rag_output_sections.json"
    )
    
    query = "이 시기의 불교 조각은 지역에 따라 다양하게 제작되었다. 처음에는 하남 하사창동의 철 조 석가 여래 좌상과 같은 대형 철불이 많이 제작되었다. 또한 덩치가 큰 석 불이 유행하였는데, 논산 관촉사 석조 미륵보살 입상이 대표적이다. 이 불상은 큰 규모에 비해 조형미는 다소 떨어지지만, 소박한 지방 문화의 모습을 잘 보여 준다. 밑줄 친 ‘이 시기’에 있었던 사실로 옳은 것은?"
    results = retriever.search(query, top_k=3)
    
    print(f"\n쿼리: {query}")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['metadata']['main_title']}] (score: {result['score']:.4f})")
        print(f"   섹션: {result['metadata']['section_title']}")
        print(f"   {result['text'][:100]}...")
"""
rag_collector.py
RAG로 외부 지식만 수집 (정답 선택 안 함!)
Pinecone 사용 버전 - 보기별 검색 지원
"""
import json
import os
import re
from typing import List, Dict, Tuple
from openai import OpenAI
from pinecone import Pinecone
from loguru import logger
import numpy as np


class RAGCollector:
    """RAG로 외부 지식 수집 (Pinecone 사용)"""
    
    def __init__(
        self,
        index_name: str = "korean-history",
        sections_file: str = "../data/rag_output_sections.json",
        query_model: str = "solar-embedding-1-large-query",
        api_key: str = None,
        pinecone_api_key: str = None
    ):
        """
        Args:
            index_name: Pinecone 인덱스 이름
            sections_file: 섹션 메타데이터 JSON 파일
            query_model: Solar 쿼리 임베딩 모델
            api_key: Upstage API 키
            pinecone_api_key: Pinecone API 키
        """
        logger.info("Initializing RAGCollector (Pinecone)...")
        
        # Upstage API 키
        if api_key:
            os.environ["UPSTAGE_API_KEY"] = api_key
        
        solar_api_key = os.getenv("UPSTAGE_API_KEY")
        if not solar_api_key:
            raise ValueError("UPSTAGE_API_KEY not found!")
        
        # Pinecone API 키
        if pinecone_api_key:
            os.environ["PINECONE_API_KEY"] = pinecone_api_key
        
        pc_api_key = os.getenv("PINECONE_API_KEY")
        if not pc_api_key:
            raise ValueError("PINECONE_API_KEY not found!")
        
        # Solar API 클라이언트
        self.solar_client = OpenAI(
            api_key=solar_api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
        self.query_model = query_model
        logger.info("✅ Solar API client initialized")
        
        # Pinecone 클라이언트
        pc = Pinecone(api_key=pc_api_key)
        self.index = pc.Index(index_name)
        logger.info(f"Pinecone index connected: {index_name}")
        
        # 섹션 메타데이터 로드
        logger.info(f"Loading sections from {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            self.sections = json.load(f)
        logger.info(f"Loaded {len(self.sections)} sections")
        
        logger.info("RAGCollector initialization complete")
    
    def extract_paragraph_items(self, paragraph: str, question: str) -> Tuple[List[Tuple[str, str]], bool]:
        """
        지문에서 보기 추출 및 참조 문제 여부 판단
        
        Returns:
            (items, is_reference_question)
            - items: [(label, text), ...] 형태
            - is_reference_question: True면 "밑줄 친 왕" 같은 참조 문제
        """
        items = []
        
        # 다양한 보기 패턴
        patterns = [
            (r'\(([가-힣])\)\s*([^\(]+?)(?=\([가-힣]\)|$)', 'parenthesis'),  # (가), (나)
            (r'([㉠-㉮])\s*([^㉠-㉮]+?)(?=[㉠-㉮]|$)', 'circle'),  # ㉠, ㉡
            (r'([ㄱ-ㅎ])\.\s*([^\n]+)', 'dot'),  # ㄱ., ㄴ.
            (r'([①-⑤])\s*([^①-⑤]+?)(?=[①-⑤]|$)', 'number_circle'),  # ①, ②
        ]
        
        for pattern, pattern_type in patterns:
            matches = re.findall(pattern, paragraph, re.DOTALL)
            if matches:
                for label, text in matches:
                    text = text.strip()
                    # 실제 내용이 있는 경우만 추가 (5자 이상)
                    if text and len(text) >= 5:
                        items.append((label, text))
                
                if items:
                    logger.debug(f"Found {len(items)} items with pattern: {pattern_type}")
                    break
        
        # 참조 문제 여부 판단
        is_reference_question = False
        
        if items:
            # 보기가 있는데 질문에서 특정 패턴으로 참조하는 경우
            # 예: "밑줄 친 (가)는?", "(가)에 들어갈 내용은?", "(가)에 해당하는 인물은?"
            reference_patterns = [
                r'밑줄.*?\([가-힣]\)',
                r'밑줄.*?[㉠-㉮]',
                r'\([가-힣]\).*?들어갈',
                r'들어갈.*?\([가-힣]\)',
                r'\([가-힣]\).*?해당',
                r'해당.*?\([가-힣]\)',
                r'\([가-힣]\).*?알맞',
                r'알맞.*?\([가-힣]\)',
                r'[㉠-㉮].*?들어갈',
                r'들어갈.*?[㉠-㉮]',
            ]
            
            for pattern in reference_patterns:
                if re.search(pattern, question):
                    is_reference_question = True
                    logger.debug(f"Reference pattern found: {pattern}")
                    break
        else:
            # 보기가 없는데 질문에 (가), (나) 등이 있으면 참조 문제
            if re.search(r'\([가-힣]\)', question) or re.search(r'[㉠-㉮]', question):
                is_reference_question = True
                logger.debug("No items but labels in question - reference question")
        
        logger.debug(f"Extracted {len(items)} items, is_reference={is_reference_question}")
        if items and len(items) <= 3:
            logger.debug(f"Sample items: {items}")
        
        return items, is_reference_question
    
    def truncate_by_sentence(self, text: str, max_length: int = 500) -> str:
        """
        텍스트를 문장 단위로 자르기
        """
        if len(text) <= max_length:
            return text
        
        # 문장 구분자
        sentence_endings = r'([.!?])\s+'
        sentences = re.split(sentence_endings, text)
        
        result = []
        current_length = 0
        i = 0
        
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                sentence = sentences[i] + sentences[i + 1]
                i += 2
            else:
                sentence = sentences[i]
                i += 1
            
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 다음 문장 추가 시 max_length 초과하면 중단
            if current_length + len(sentence) > max_length:
                break
            
            result.append(sentence)
            current_length += len(sentence) + 1
        
        # 결과가 비어있으면 원본의 max_length만큼 반환
        if not result:
            return text[:max_length]
        
        return ' '.join(result)
    
    def collect_facts(
        self,
        paragraph: str,
        question: str,
        choices: List[str] = None,
        top_k: int = 5,
        use_choice_search: bool = True,
        use_item_search: bool = True,
        max_fact_length: int = 500
    ) -> Dict:
        """
        문제 관련 사실 수집
        
        검색 전략:
        - 각 보기/선택지에서 가장 관련성 높은 것만 1개씩 선택
        """
        all_results = []
        search_method = "paragraph_question"
        paragraph_items = []
        is_reference_question = False
        
        # Step 1: 보기 추출 및 참조 문제 판단
        if use_item_search:
            paragraph_items, is_reference_question = self.extract_paragraph_items(paragraph, question)
        
        # Step 2-A: 참조 문제
        if is_reference_question:
            search_method = "reference_question"
            logger.info("Detected reference question - using paragraph+question only")
            
            query_main = f"{paragraph} {question}"
            results = self._search(query_main, top_k=top_k)
            
            for result in results:
                result['search_type'] = 'main'
                all_results.append(result)
        
        # Step 2-B: 보기 문제 (보기별 + 선택지별 각 1개씩)
        elif paragraph_items:
            search_method = "with_items_and_choices"
            num_items = len(paragraph_items)
            logger.info(f"Found {num_items} items in paragraph")
            
            # 각 보기별로 가장 높은 점수 1개씩만
            for label, item_text in paragraph_items:
                query_item = f"{question} {item_text}"
                results_item = self._search(query_item, top_k=1)  # 1개만!
                
                if results_item:
                    result = results_item[0]
                    result['related_to'] = f"({label})"
                    result['search_type'] = 'paragraph_item'
                    all_results.append(result)
            
            # 각 선택지별로 가장 높은 점수 1개씩만
            meaningful_choice_count = 0
            if use_choice_search and choices:
                for i, choice in enumerate(choices, 1):
                    if self._is_meaningful_choice(choice):
                        query_choice = f"{question} {choice}"
                        results_choice = self._search(query_choice, top_k=1)  # 1개만!
                        
                        if results_choice:
                            result = results_choice[0]
                            result['related_to'] = f"선택지{i}"
                            result['search_type'] = 'choice'
                            all_results.append(result)
                            meaningful_choice_count += 1
                
                if meaningful_choice_count > 0:
                    logger.info(f"Also searched {meaningful_choice_count} meaningful choices")
            
            # 기본 검색 1개
            query_main = f"{paragraph} {question}"
            results_main = self._search(query_main, top_k=1)  # 1개만!
            if results_main:
                result = results_main[0]
                result['search_type'] = 'main'
                all_results.append(result)
        
        # Step 2-C: 선택지별 검색 (보기 없음)
        elif use_choice_search and choices:
            search_method = "with_choices"
            
            # 각 선택지별로 가장 높은 점수 1개씩만
            meaningful_choice_count = 0
            for i, choice in enumerate(choices, 1):
                if self._is_meaningful_choice(choice):
                    query_choice = f"{paragraph} {question} {choice}"
                    results_choice = self._search(query_choice, top_k=1)  # 1개만!
                    
                    if results_choice:
                        result = results_choice[0]
                        result['related_to'] = f"선택지{i}"
                        result['search_type'] = 'choice'
                        all_results.append(result)
                        meaningful_choice_count += 1
            
            logger.info(f"Searched {meaningful_choice_count} meaningful choices")
            
            # 기본 검색 1개
            if meaningful_choice_count > 0:
                query_main = f"{paragraph} {question}"
                results_main = self._search(query_main, top_k=1)  # 1개만!
                if results_main:
                    result = results_main[0]
                    result['search_type'] = 'main'
                    all_results.append(result)
        
        # Step 2-D: 기본 검색만
        else:
            query_main = f"{paragraph} {question}"
            results = self._search(query_main, top_k=top_k)
            for result in results:
                result['search_type'] = 'main'
                all_results.append(result)
        
        # Step 3: 중복 제거 (정렬 안 함! 순서 유지)
        unique_results = []
        seen_texts = set()
        
        for doc in all_results:
            text = doc['text']
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(doc)
        
        # 순서 그대로 유지 (정렬 안 함!)
        top_results = unique_results
        
        # Step 4: 대상 추출
        target = ""
        if top_results:
            # 가장 높은 점수의 결과에서 추출
            top_result = max(top_results, key=lambda x: x['score'])
            target = top_result.get('metadata', {}).get('main_title', '')
            target = re.sub(r'\s*\(.*?\)', '', target).strip()
        
        # Step 5: 사실 및 소스 수집
        facts = []
        sources = []
        
        for doc in top_results:
            # 문장 단위로 자르기
            fact_text = self.truncate_by_sentence(doc['text'], max_fact_length)
            facts.append(fact_text)
            
            source = {
                "title": doc.get('metadata', {}).get('main_title', ''),
                "sub_title": doc.get('metadata', {}).get('sub_title', ''),
                "section_title": doc.get('metadata', {}).get('section_title', ''),
                "score": float(doc.get('score', 0)),
                "original_length": len(doc['text']),
                "truncated_length": len(fact_text)
            }
            
            if 'related_to' in doc:
                source['related_to'] = doc['related_to']
            if 'search_type' in doc:
                source['search_type'] = doc['search_type']
            
            sources.append(source)
        
        return {
            "target": target,
            "facts": facts,
            "sources": sources,
            "search_method": search_method,
            "is_reference_question": is_reference_question,
            "paragraph_items": [(label, text[:100]) for label, text in paragraph_items] if paragraph_items else []
        }
    
    def _is_meaningful_choice(self, choice: str) -> bool:
        """
        선택지가 의미있는 내용인지 판단
        
        의미 없는 선택지 예시:
        - "1 - (가)-(나)-(다)-(라)"
        - "2 - ㄱ, ㄴ, ㄷ"
        - "3 - ①, ②, ③"
        
        의미 있는 선택지 예시:
        - "1 - 신문고를 설치하였다."
        - "2 - 집현전을 설치하였다."
        """
        # 선택지 번호 제거 ("1 - " 같은 부분)
        clean_choice = re.sub(r'^\d+\s*-\s*', '', choice).strip()
        
        # 길이가 너무 짧으면 의미 없음
        if len(clean_choice) < 10:
            return False
        
        # 보기 참조만 있는 패턴 (의미 없음)
        reference_only_patterns = [
            r'^[\(가-힣\)]+[-,\s]*[\(가-힣\)]*[-,\s]*[\(가-힣\)]*$',  # (가)-(나)-(다)
            r'^[ㄱ-ㅎ]+[,\s]*[ㄱ-ㅎ]*[,\s]*[ㄱ-ㅎ]*$',  # ㄱ, ㄴ, ㄷ
            r'^[①-⑤]+[,\s]*[①-⑤]*[,\s]*[①-⑤]*$',  # ①, ②, ③
            r'^[㉠-㉮]+[,\s]*[㉠-㉮]*[,\s]*[㉠-㉮]*$',  # ㉠, ㉡, ㉢
        ]
        
        for pattern in reference_only_patterns:
            if re.match(pattern, clean_choice):
                return False
        
        # 실제 내용이 있음
        return True

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """중복 제거 및 점수순 정렬"""
        unique_results = []
        seen_texts = set()
        
        for doc in results:
            text = doc['text']
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(doc)
        
        # 점수순 정렬
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        
        return unique_results
    
    def _get_embedding(self, text: str, retry_count: int = 3) -> List[float]:
        """Solar 임베딩 (재시도 로직 포함)"""
        import time
        
        for attempt in range(retry_count):
            try:
                response = self.solar_client.embeddings.create(
                    input=[text],
                    model=self.query_model
                )
                return response.data[0].embedding
            except Exception as e:
                if "429" in str(e) and attempt < retry_count - 1:
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Embedding failed: {e}")
                    raise
    
    def _search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Pinecone 검색"""
        # 쿼리 임베딩
        query_emb = self._get_embedding(query)
        
        # Pinecone 검색
        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True,
            namespace=""
        )
        
        # 결과 구성
        output = []
        for match in results.matches:
            section_idx = match.metadata.get("section_idx", 0)
            
            if section_idx < len(self.sections):
                section = self.sections[section_idx]
                output.append({
                    "text": section['text'],
                    "score": float(match.score),
                    "metadata": {
                        "main_title": section['title'],
                        "sub_title": section['sub_title'],
                        "section_title": section['section_title']
                    }
                })
        
        return output
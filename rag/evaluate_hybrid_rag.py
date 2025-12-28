"""
3단계 RAG

Step 1: 지문 검색 → 첫 번째 제목
Step 2: 제목을 질문 앞에 붙이기
Step 3: 각 선지 검색
"""
import json
import sys
import os
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
from loguru import logger
import re
import numpy as np
from openai import OpenAI

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from sentence_transformers import SentenceTransformer
from retrieval_dense import DenseRetrieval
from retrieval_bm25 import BM25Retrieval
from reranker import Reranker


class ThreeStepRAG:
    """3단계 RAG"""
    
    def __init__(
        self,
        dense_index_path="models/index_solar",
        bm25_index_path="models/bm25_index.pkl",
        embedding_model="solar-embedding-1-large-query",
        use_reranker=True,
        api_key=None  # 추가
    ):
        # API 키 설정 (인자 우선, 없으면 환경변수)
        if api_key:
            os.environ["UPSTAGE_API_KEY"] = api_key
        
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "UPSTAGE_API_KEY not found!\n"
                "Set: export UPSTAGE_API_KEY='your-key' or use --api-key option"
            )
        
        # Solar API 클라이언트 초기화
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
        self.embedding_model_name = embedding_model
        self.embedding_dim = 4096  # Solar embedding 차원
        
        # Dense retriever 로드
        logger.info(f"Dense 인덱스 로드: {dense_index_path}")
        self.dense_retriever = DenseRetrieval(embedding_dim=self.embedding_dim)
        self.dense_retriever.load(dense_index_path)
        logger.info(f"Dense: {self.dense_retriever.index.ntotal}개 문서")
        
        # BM25
        self.bm25_retriever = None
        if os.path.exists(bm25_index_path):
            logger.info(f"BM25 인덱스 로드: {bm25_index_path}")
            self.bm25_retriever = BM25Retrieval()
            self.bm25_retriever.load(bm25_index_path)
            logger.info(f"BM25: {len(self.bm25_retriever.corpus)}개 문서")
        
        # Reranker
        self.reranker = None
        if use_reranker:
            logger.info("Reranker 로드 중...")
            self.reranker = Reranker()
            logger.info("Reranker 로드 완료")
        
        logger.info("초기화 완료")
    
    def load_data(self, data_path: str, topic: str = "한국사") -> List[Dict]:
        logger.info(f"데이터 로드: {data_path}")
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        logger.info(f"전체 데이터: {len(data)}개")
        
        if 'topic' in data[0]:
            filtered_data = [item for item in data if item.get('topic') == topic]
            logger.info(f"'{topic}' 필터링 후: {len(filtered_data)}개")
        else:
            filtered_data = data
        
        return filtered_data
    
    def rrf_merge(self, dense_results, bm25_results, k=60):
        """RRF"""
        doc_scores = {}
        
        for rank, doc in enumerate(dense_results, start=1):
            doc_text = doc['text']
            if doc_text not in doc_scores:
                doc_scores[doc_text] = {
                    'text': doc_text,
                    'metadata': doc.get('metadata', {}),
                    'score': 0.0
                }
            doc_scores[doc_text]['score'] += 1.0 / (k + rank)
        
        if bm25_results:
            for rank, doc in enumerate(bm25_results, start=1):
                doc_text = doc['text']
                if doc_text not in doc_scores:
                    doc_scores[doc_text] = {
                        'text': doc_text,
                        'metadata': doc.get('metadata', {}),
                        'score': 0.0
                    }
                doc_scores[doc_text]['score'] += 1.0 / (k + rank)
        
        return sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
    
    def encode_query(self, query: str, retry_count=3) -> np.ndarray:
        """Solar API로 쿼리 임베딩 (재시도 로직 포함)"""
        import time
        
        for attempt in range(retry_count):
            try:
                response = self.client.embeddings.create(
                    input=[query],
                    model=self.embedding_model_name
                )
                embedding = np.array(response.data[0].embedding)
                # 정규화
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            except Exception as e:
                if "429" in str(e) and attempt < retry_count - 1:
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"쿼리 임베딩 실패: {e}")
                    raise
    
    def hybrid_search(self, query: str, top_k=10, use_reranker=True):
        """Hybrid Search: Dense + BM25 + Reranker"""
        # Dense - Solar API 사용
        query_emb = self.encode_query(query)  # 수정된 부분
        dense_results = self.dense_retriever.search(query_emb, top_k=100)
        
        # BM25
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query, top_k=100)
        
        # RRF
        hybrid = self.rrf_merge(dense_results, bm25_results)
        
        # Reranker
        if use_reranker and self.reranker and len(hybrid) > 0:
            candidates = hybrid[:50]
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return reranked
        
        return hybrid[:top_k]
    
    def solve(
        self,
        paragraph: str,
        question: str,
        choices: List[str]
    ) -> Tuple[int, Dict]:
        """
        3단계 RAG
        
        Step 1: 지문 검색 → 첫 번째 제목
        Step 2: 제목 + 질문
        Step 3: 각 선지 검색
        """
        # Step 1: 지문 + 질문으로 검색
        query_step1 = f"{paragraph} {question}"
        results_step1 = self.hybrid_search(query_step1, top_k=10, use_reranker=True)
        
        # 첫 번째 문서 제목 추출
        prefix = ""
        resolve_docs = []
        if results_step1:
            top_doc = results_step1[0]
            title = top_doc.get('metadata', {}).get('main_title', '')
            
            if title:
                # 괄호 제거
                clean_title = re.sub(r'\s*\(.*?\)', '', title).strip()
                prefix = clean_title
                
                # 상세 정보 저장
                resolve_docs = [
                    {
                        "rank": i+1,
                        "title": doc.get('metadata', {}).get('main_title', ''),
                        "score": float(doc.get('score', 0)),
                        "text_preview": doc.get('text', '')[:100]
                    }
                    for i, doc in enumerate(results_step1[:10])
                ]
                
                # Step 1 로깅
                logger.debug(f"\n{'='*80}")
                logger.debug(f"[Step 1] 참조 해결")
                logger.debug(f"쿼리: {query_step1[:100]}...")
                logger.debug(f"추출된 Prefix: {prefix}")
                logger.debug(f"\n상위 5개 검색 결과:")
                for doc in resolve_docs[:5]:
                    logger.debug(f"  {doc['rank']}. [{doc['title']}] (score: {doc['score']:.4f})")
                    logger.debug(f"     {doc['text_preview']}...")
                logger.debug(f"{'='*80}\n")
        
        # Step 2 & 3: 각 선지 검색
        choice_scores = []
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"[Step 2 & 3] 선지별 검색")
        logger.debug(f"Prefix 사용: {prefix if prefix else '(없음)'}")
        logger.debug(f"{'='*80}\n")
        
        for idx, choice in enumerate(choices):
            # 쿼리: (제목) + 질문 + 선지
            if prefix:
                query = f"{prefix} {question} {choice}"
            else:
                query = f"{question} {choice}"
            
            logger.debug(f"\n--- 선지 {idx+1} ---")
            logger.debug(f"선지: {choice}")
            logger.debug(f"쿼리: {query[:100]}...")
            
            # Hybrid 검색
            results = self.hybrid_search(query, top_k=5, use_reranker=True)
            
            # 검색 결과 로깅
            logger.debug(f"검색 결과:")
            for i, r in enumerate(results[:3], 1):
                logger.debug(f"  {i}. [{r.get('metadata', {}).get('main_title', '')}] (score: {r.get('score', 0):.4f})")
                logger.debug(f"     {r.get('text', '')[:80]}...")
            
            # 점수 계산
            if results:
                avg_score = sum(r['score'] for r in results) / len(results)
                max_score = results[0]['score']
                final_score = 0.6 * avg_score + 0.4 * max_score
            else:
                avg_score = max_score = final_score = 0.0
            
            logger.debug(f"점수: avg={avg_score:.4f}, max={max_score:.4f}, final={final_score:.4f}")
            
            choice_scores.append({
                "choice_idx": idx,
                "choice": choice,
                "score": final_score,
                "avg_score": avg_score,
                "max_score": max_score,
                "top_doc": results[0].get('metadata', {}).get('main_title', '') if results else '',
                "search_results": [
                    {
                        "rank": i+1,
                        "title": r.get('metadata', {}).get('main_title', ''),
                        "score": float(r.get('score', 0)),
                        "text_preview": r.get('text', '')[:100]
                    }
                    for i, r in enumerate(results[:5])
                ]
            })
        
        # 최고 점수 선택
        predicted_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i]['score'])
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"[최종 결과]")
        logger.debug(f"예측: 선지 {predicted_idx + 1}")
        logger.debug(f"점수 비교:")
        for cs in choice_scores:
            marker = "✅" if cs['choice_idx'] == predicted_idx else "  "
            logger.debug(f"{marker} 선지 {cs['choice_idx']+1}: {cs['score']:.4f} - {cs['choice'][:50]}...")
        logger.debug(f"{'='*80}\n")
        
        detail = {
            "method": "three_step_rag",
            "prefix": prefix,
            "resolve_docs": resolve_docs,
            "choice_scores": choice_scores
        }
        
        return predicted_idx, detail
    
    def evaluate(
        self,
        data: List[Dict],
        output_path: str = "results_three_step.json"
    ) -> Dict:
        """전체 평가"""
        results = []
        correct = 0
        total = len(data)
        
        logger.info(f"3단계 RAG 평가 시작: {total}개 문제")
        
        for idx, item in enumerate(tqdm(data, desc="평가 진행")):
            try:
                paragraph = item.get('paragraph', '')
                question = item.get('question', '')
                choices_str = item.get('choices', '[]')
                
                if isinstance(choices_str, str):
                    import ast
                    try:
                        choices = ast.literal_eval(choices_str)
                    except:
                        choices = [choices_str]
                else:
                    choices = choices_str
                
                answer_idx = int(item.get('answer', 0))
                
                # 문제 풀이
                predicted_idx, detail = self.solve(paragraph, question, choices)
                
                # 정답 확인
                is_correct = (predicted_idx == answer_idx)
                if is_correct:
                    correct += 1
                
                result = {
                    "index": idx,
                    "paragraph": paragraph[:100] + "...",
                    "question": question,
                    "choices": choices,
                    "answer": answer_idx,
                    "predicted": predicted_idx,
                    "is_correct": is_correct,
                    "detail": detail
                }
                results.append(result)
                
                # 중간 로그
                if (idx + 1) % 10 == 0:
                    accuracy = correct / (idx + 1)
                    logger.info(f"진행: {idx+1}/{total}, 정확도: {accuracy:.2%}")
                
            except Exception as e:
                logger.error(f"문제 {idx} 처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        accuracy = correct / total if total > 0 else 0
        
        evaluation_result = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        logger.info("="*80)
        logger.info(f"3단계 RAG 평가 완료!")
        logger.info(f"   총 문제: {total}개")
        logger.info(f"   정답: {correct}개")
        logger.info(f"   정확도: {accuracy:.2%}")
        logger.info(f"   결과 저장: {output_path}")
        logger.info("="*80)
        
        return evaluation_result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="3단계 RAG")
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--topic', type=str, default='한국사')
    parser.add_argument('--dense-index', type=str, default='models/index_solar')
    parser.add_argument('--bm25-index', type=str, default='models/bm25_index.pkl')
    parser.add_argument('--output', type=str, default='results_three_step.json')
    parser.add_argument('--no-reranker', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='상세 로깅 (DEBUG 레벨)')
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Upstage API key (or set UPSTAGE_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    
    system = ThreeStepRAG(
        dense_index_path=args.dense_index,
        bm25_index_path=args.bm25_index,
        use_reranker=not args.no_reranker,
        api_key=args.api_key  
    )
    
    data = system.load_data(args.data_path, topic=args.topic)
    results = system.evaluate(data=data, output_path=args.output)


if __name__ == "__main__":
    main()
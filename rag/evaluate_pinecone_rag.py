"""
evaluate_pinecone_rag.py
2-Stage Retrieval with Pinecone (+ Optional Classifier)
1차: paragraph + question → 관련 문서 찾기
2차: 1차 결과 + question → 최종 답변 찾기
"""
import os
import json
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm
import pandas as pd
import torch

try:
    from pinecone import Pinecone
    from openai import OpenAI
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    raise


class KoreanHistoryClassifier:
    """한국사 분류기 (Optional)"""
    
    def __init__(self, checkpoint_path: str):
        logger.info(f"Loading classifier from {checkpoint_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        
        # GPU 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Classifier loaded on {self.device}")
    
    def predict(self, paragraph: str, question: str, choices: List[str]) -> Dict:
        """한국사 여부 예측"""
        # 입력 구성
        choices_text = " ".join([f"({i+1}) {c}" for i, c in enumerate(choices)])
        input_text = f"{paragraph} {question} {choices_text}"
        
        # 토크나이징
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        # 라벨 매핑
        is_korean_history = (predicted_class == 1)
        label = "한국사" if is_korean_history else "비한국사"
        
        return {
            "is_korean_history": is_korean_history,
            "probability": confidence,
            "label": label
        }


class TwoStageRetrieval:
    """2-Stage Retrieval"""
    
    def __init__(
        self,
        index_name: str = "korean-history",
        sections_file: str = "../data/rag_output_sections.json",
        query_model: str = "solar-embedding-1-large-query"
    ):
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
        self.query_model = query_model
        
        # Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
        
        # 섹션 메타데이터
        logger.info(f"Loading sections from {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            self.sections = json.load(f)
        
        logger.info(f"✅ Loaded {len(self.sections)} sections")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Solar 임베딩"""
        response = self.solar_client.embeddings.create(
            input=[text],
            model=self.query_model
        )
        return response.data[0].embedding
    
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
    
    def retrieve_two_stage(
        self,
        paragraph: str,
        question: str,
        top_k_stage1: int = 10,
        top_k_stage2: int = 5
    ) -> Dict:
        """
        2-Stage Retrieval with Smart Query Reformulation
        
        Stage 1: 찾아야 할 대상 명시 + paragraph + question → 관련 문서 찾기
        Stage 2: 추출된 키워드 + question → 더 정확한 답변 찾기
        """
        logger.info("=" * 80)
        logger.info("2-Stage Retrieval")
        logger.info("=" * 80)
        
        # 질문에서 찾아야 할 대상 파악
        search_target = self._detect_search_target(question)
        logger.info(f"Search target: {search_target}")
        
        # Stage 1: 대상 명시 + paragraph + question으로 검색
        logger.info("Stage 1: Searching with target hint + paragraph + question")
        if search_target:
            query_stage1 = f"{search_target}를 찾아라. {paragraph} {question}"
        else:
            query_stage1 = f"{paragraph} {question}"
        
        stage1_results = self._search(query_stage1, top_k=top_k_stage1)
        
        if not stage1_results:
            logger.warning("No results in Stage 1")
            return {
                "stage1_results": [],
                "stage1_top1": None,
                "stage2_query": None,
                "stage2_results": [],
                "final_answer": None
            }
        
        # Stage 1의 top-1 결과
        stage1_top1 = stage1_results[0]
        logger.info(f"Stage 1 Top-1: [{stage1_top1['metadata']['main_title']}] (score: {stage1_top1['score']:.4f})")
        
        # 스마트 쿼리 생성 (Stage 2용)
        stage2_query = self._create_smart_query(stage1_top1, question)
        logger.info(f"Stage 2 Query: {stage2_query[:100]}...")
        
        # Stage 2: 스마트 쿼리로 다시 검색
        logger.info("Stage 2: Searching with smart query")
        stage2_results = self._search(stage2_query, top_k=top_k_stage2)
        
        # 최종 답변 (Stage 2의 top-1)
        final_answer = stage2_results[0] if stage2_results else None
        
        if final_answer:
            logger.info(f"Final Answer: [{final_answer['metadata']['main_title']}] (score: {final_answer['score']:.4f})")
        
        logger.info("=" * 80)
        
        return {
            "stage1_results": stage1_results,
            "stage1_top1": stage1_top1,
            "stage2_query": stage2_query,
            "stage2_results": stage2_results,
            "final_answer": final_answer
        }
    
    def _detect_search_target(self, question: str) -> str:
        """
        질문에서 찾아야 할 대상 파악
        
        Returns:
            "왕", "나라", "인물", "지역", "단체", "사건", etc.
        """
        # 왕 (재위 기간, 국왕, 왕의 업적 등)
        if any(kw in question for kw in ['왕의 재위', '왕의 업적', '국왕', '왕대', '밑줄 친 \'왕\'', '밑줄 친 왕']):
            return "왕"
        
        # 나라/국가
        if any(kw in question for kw in ['나라', '국가']):
            return "나라"
        
        # 인물 (왕 제외)
        if any(kw in question for kw in ['인물', '승려', '밑줄 친 \'그\'', '밑줄 친 그', '밑줄 친 \'저\'', '이끈 인물', '주장한 인물']):
            return "인물"
        
        # 지역/장소
        if any(kw in question for kw in ['지역', '이곳', '장소']):
            return "지역"
        
        # 단체/조직
        if any(kw in question for kw in ['단체', '조직']):
            return "단체"
        
        # 사건/운동 (순서 문제 제외)
        if '순으로' not in question:
            if any(kw in question for kw in ['운동', '사건']):
                return "사건"
        
        # 제도/정책/법령/조약
        if any(kw in question for kw in ['법령', '조약', '정책', '제도', '개헌안']):
            return "제도"
        
        # 문화재/유산
        if any(kw in question for kw in ['문화유산', '역사서', '유적', '유물']):
            return "문화재"
        
        # 시기/시대 (다른 것과 겹치지 않을 때만)
        if any(kw in question for kw in ['시기', '시대']):
            # "왕의 재위 기간"처럼 이미 다른 카테고리에 속하면 제외
            if '왕' not in question and '국왕' not in question:
                return "시기"
        
        return ""  # 특정 대상 없음
    
    def _create_smart_query(self, doc: Dict, question: str) -> str:
        """
        Stage 1 결과 + 질문 유형 분석 → Stage 2 쿼리 생성
        
        핵심: "뭘 찾아야 하는지" 명확히 전달!
        """
        import re
        
        title = doc['metadata']['main_title']
        text = doc['text']
        
        # 질문 유형 분석
        answer_type = self._detect_answer_type(question)
        logger.info(f"Answer type detected: {answer_type}")
        
        # 연도 추출 (title에서)
        year_match = re.search(r'연도[:：]\s*(\d+)', title)
        century_match = re.search(r'(\d+)세기', title)
        period_match = re.search(r'(고려|조선|신라|고구려|백제|발해)', title)
        
        # 유형별 쿼리 생성
        if answer_type == "시기":
            # "이 시기", "재위 기간" 등 → 시기 정보 찾아라!
            keywords = []
            
            # 연도 → 세기
            if year_match:
                year = int(year_match.group(1))
                century = (year - 1) // 100 + 1
                keywords.append(f"{century}세기")
            
            if century_match:
                keywords.append(f"{century_match.group(1)}세기")
            
            if period_match:
                keywords.append(period_match.group(1))
            
            # Stage 2 쿼리: "시기를 찾아라" 명시!
            if keywords:
                query = f"시기: {' '.join(keywords)}. {question}"
            else:
                query = f"시기 정보를 찾아라. {text[:300]} {question}"
            
            return query
        
        elif answer_type == "인물":
            # "이 인물", "누구" → 인물 이름 찾아라!
            # title이 보통 인물명
            person_match = re.search(r'^([가-힣]+)\s*\(', title)
            if person_match:
                person = person_match.group(1)
                query = f"인물: {person}. {question}"
            else:
                query = f"인물 이름을 찾아라. {text[:300]} {question}"
            
            return query
        
        elif answer_type == "장소":
            # "이곳", "어디" → 지명 찾아라!
            # 구체적 장소 추출 시도하지 않고, 타입만 명시
            query = f"장소를 찾아라. {text[:400]} {question}"
            return query
        
        elif answer_type == "사건":
            # "이 운동", "이 사건" → 사건명 찾아라!
            # 구체적 사건명 추출 시도하지 않고, 타입만 명시
            query = f"사건을 찾아라. {text[:400]} {question}"
            return query
        
        elif answer_type == "활동/업적":
            # "활동", "업적" → 구체적 행위 찾아라!
            query = f"활동과 업적을 찾아라. {text[:300]} {question}"
            return query
        
        else:
            # 기본: text + question
            return f"{text[:500]} {question}"
    
    def _detect_answer_type(self, question: str) -> str:
        """
        질문에서 찾아야 할 답변 유형 감지
        
        Returns:
            "시기", "인물", "장소", "사건", "활동/업적", "일반"
        """
        # 시기 관련
        if any(kw in question for kw in ['시기', '시대', '재위 기간', '때', '언제']):
            return "시기"
        
        # 인물 관련
        if any(kw in question for kw in ['인물', '누구', '왕', '그']):
            return "인물"
        
        # 장소 관련
        if any(kw in question for kw in ['이곳', '지역', '어디']):
            return "장소"
        
        # 사건 관련
        if any(kw in question for kw in ['운동', '사건', '전쟁', '난']):
            return "사건"
        
        # 활동/업적 관련
        if any(kw in question for kw in ['활동', '업적', '설명으로 옳은']):
            return "활동/업적"
        
        return "일반"


def evaluate_two_stage(
    data_path: str,
    index_name: str = "korean-history",
    sections_file: str = "../data/rag_output_sections.json",
    output_path: str = "results_two_stage.json",
    classifier_path: Optional[str] = None  # 분류기 경로 (옵션)
):
    """2-Stage Retrieval 평가 (+ Optional Classifier)"""
    logger.info("=" * 80)
    logger.info("2-Stage Retrieval Evaluation")
    logger.info("=" * 80)
    
    # 분류기 로드 (옵션)
    classifier = None
    if classifier_path:
        logger.info(f"Classifier enabled: {classifier_path}")
        classifier = KoreanHistoryClassifier(checkpoint_path=classifier_path)
    else:
        logger.info("Classifier disabled - processing all questions")
    
    # Retriever 초기화
    retriever = TwoStageRetrieval(
        index_name=index_name,
        sections_file=sections_file
    )
    
    # 데이터 로드
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} questions")
    
    # topic=='한국사'만 필터링
    if 'topic' in df.columns:
        df_korean = df[df['topic'] == '한국사'].copy()
        logger.info(f"Filtered to Korean history only: {len(df_korean)} questions (from {len(df)} total)")
        df = df_korean
    else:
        logger.warning("'topic' column not found - processing all questions")
    
    # 평가
    results = []
    korean_history_count = 0
    non_korean_history_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        paragraph = row['paragraph']
        question = row['question']
        choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
        answer = row['answer']
        
        result = {
            "id": row.get('id', idx),
            "paragraph": paragraph,
            "question": question,
            "choices": choices,
            "answer": answer
        }
        
        # 분류기 사용 (있으면)
        if classifier:
            classification = classifier.predict(paragraph, question, choices)
            result["classification"] = classification
            
            # 비한국사면 RAG 스킵
            if not classification['is_korean_history']:
                non_korean_history_count += 1
                result["rag_result"] = None
                results.append(result)
                continue
            
            korean_history_count += 1
        
        # 2-Stage Retrieval
        rag_result = retriever.retrieve_two_stage(
            paragraph=paragraph,
            question=question,
            top_k_stage1=10,
            top_k_stage2=5
        )
        
        # 결과 저장
        result["rag_result"] = {
            "stage1_top1": {
                "title": rag_result['stage1_top1']['metadata']['main_title'] if rag_result['stage1_top1'] else None,
                "text": rag_result['stage1_top1']['text'][:200] if rag_result['stage1_top1'] else None,
                "score": rag_result['stage1_top1']['score'] if rag_result['stage1_top1'] else None
            },
            "stage2_query": rag_result.get('stage2_query', ''),
            "final_answer": {
                "title": rag_result['final_answer']['metadata']['main_title'] if rag_result['final_answer'] else None,
                "text": rag_result['final_answer']['text'][:200] if rag_result['final_answer'] else None,
                "score": rag_result['final_answer']['score'] if rag_result['final_answer'] else None
            }
        }
        
        results.append(result)
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 통계
    logger.info("=" * 80)
    logger.info(f"Total questions: {len(df)}")
    if classifier:
        logger.info(f"Korean history: {korean_history_count} ({korean_history_count/len(df)*100:.1f}%)")
        logger.info(f"Non-Korean history: {non_korean_history_count} ({non_korean_history_count/len(df)*100:.1f}%)")
    logger.info(f"Results saved to {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--index-name', type=str, default='korean-history')
    parser.add_argument('--sections-file', type=str, default='../data/rag_output_sections.json')
    parser.add_argument('--output', type=str, default='results_two_stage.json')
    parser.add_argument('--classifier-path', type=str, default=None, help='Classifier checkpoint path (optional)')
    
    args = parser.parse_args()
    
    evaluate_two_stage(
        data_path=args.data_path,
        index_name=args.index_name,
        sections_file=args.sections_file,
        output_path=args.output,
        classifier_path=args.classifier_path
    )
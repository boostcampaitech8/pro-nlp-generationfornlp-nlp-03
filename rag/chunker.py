"""
chunker.py
데이터 청킹 모듈
텍스트를 문장 단위로 나누어 처리하는 기능 제공
"""
import json
import pickle
import re
from typing import Dict, List, Tuple
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
from tqdm import tqdm


class DataChunker:
    """텍스트 데이터를 문장 단위로 청킹하는 클래스"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 1,  # 문장 단위 오버랩
        tokenizer_name: str = "monologg/kobert"
    ):
        """
        Args:
            chunk_size: 청크의 최대 토큰 수
            chunk_overlap: 청크 간 중복 문장 수
            tokenizer_name: 사용할 토크나이저 이름
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            trust_remote_code=True
        )
        
    def split_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분리
        
        Args:
            text: 입력 텍스트
            
        Returns:
            문장 리스트
        """
        # 마침표(.), 물음표(?), 느낌표(!) 뒤에 공백이나 줄바꿈이 오는 경우
        sentence_endings = r'([.!?])\s+'
        
        # 문장 분리
        sentences = re.split(sentence_endings, text)
        
        # 분리된 부호와 문장 재결합
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                result.append(sentences[i] + sentences[i + 1])
                i += 2
            else:
                if sentences[i].strip():
                    result.append(sentences[i])
                i += 1
        
        # 빈 문장 제거 및 공백 정리
        result = [s.strip() for s in result if s.strip()]
        
        return result
        
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        텍스트를 문장 단위로 청크 분할
        
        Args:
            text: 분할할 텍스트
            metadata: 청크에 포함할 메타데이터
            
        Returns:
            청크 딕셔너리 리스트
        """
        # 텍스트를 문장으로 분리
        sentences = self.split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        
        for sentence in sentences:
            # 문장 토큰 수 계산
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_token_count = len(sentence_tokens)
            
            # 현재 청크에 추가했을 때 크기 초과 여부 확인
            if current_token_count + sentence_token_count > self.chunk_size:
                # 현재 청크가 비어있지 않으면 저장
                if current_chunk_sentences:
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunk = {
                        "text": chunk_text,
                        "sentence_count": len(current_chunk_sentences),
                        "token_count": current_token_count,
                    }
                    
                    # 메타데이터 추가
                    if metadata:
                        chunk.update(metadata)
                    
                    chunks.append(chunk)
                    
                    # 오버랩 처리: 마지막 N개 문장 유지
                    if self.chunk_overlap > 0 and len(current_chunk_sentences) > self.chunk_overlap:
                        overlap_sentences = current_chunk_sentences[-self.chunk_overlap:]
                        overlap_tokens = sum(
                            len(self.tokenizer.encode(s, add_special_tokens=False))
                            for s in overlap_sentences
                        )
                        current_chunk_sentences = overlap_sentences
                        current_token_count = overlap_tokens
                    else:
                        current_chunk_sentences = []
                        current_token_count = 0
            
            # 문장 추가
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_token_count
        
        # 마지막 청크 처리
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = {
                "text": chunk_text,
                "sentence_count": len(current_chunk_sentences),
                "token_count": current_token_count,
            }
            
            if metadata:
                chunk.update(metadata)
            
            chunks.append(chunk)
                
        return chunks
    
    def chunk_documents(
        self, 
        documents: List[Dict],
        text_key: str = "text",
        title_key: str = "title"
    ) -> List[Dict]:
        """
        여러 문서를 문장 단위로 청크 분할
        
        Args:
            documents: 문서 리스트
            text_key: 텍스트 필드 키
            title_key: 제목 필드 키
            
        Returns:
            모든 청크 리스트
        """
        all_chunks = []
        # Modify
        for doc_id, doc in enumerate(tqdm(documents, desc="Chunking documents")):
            if "content" in doc and isinstance(doc["content"], list):
                text_parts = []
                for section in doc["content"]:
                    section_title = section.get("section_title", "")
                    section_text = section.get("section_text", "")
                    text_parts.append(f"{section_title}\n{section_text}")
                text = "\n\n".join(text_parts)
            else:
                text = doc.get(text_key, "")
            
            title = doc.get(title_key, "")
            
            full_text = f"{title}\n{text}" if title else text
            
            # 메타데이터 준비 (원본 문서의 메타데이터 포함)
            metadata = {
                "doc_id": doc_id,
                "title": title,
            }
            
            # 원본 문서에 metadata가 있으면 병합
            if "metadata" in doc:
                metadata.update(doc["metadata"])
            
            # 청크 생성
            chunks = self.chunk_text(full_text, metadata)
            
            # 청크 ID 추가
            for chunk_id, chunk in enumerate(chunks):
                chunk["chunk_id"] = chunk_id
            
            all_chunks.extend(chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # 통계 출력
        avg_sentences = sum(c.get('sentence_count', 0) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        avg_tokens = sum(c.get('token_count', 0) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        
        logger.info(f"Average sentences per chunk: {avg_sentences:.1f}")
        logger.info(f"Average tokens per chunk: {avg_tokens:.1f}")
        
        return all_chunks

    
    def save_chunks(self, chunks: List[Dict], output_path: str):
        """청크를 파일로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(chunks, f)
            
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        
    def load_chunks(self, input_path: str) -> List[Dict]:
        """파일에서 청크 로드"""
        with open(input_path, 'rb') as f:
            chunks = pickle.load(f)
            
        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks


def process_korean_history_data(
    raw_data_path: str,
    output_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 1  # 문장 단위 오버랩
):
    """
    한국사 데이터를 처리하고 문장 단위로 청킹
    
    Args:
        raw_data_path: 원본 데이터 경로
        output_path: 처리된 데이터 저장 경로
        chunk_size: 청크 최대 토큰 수
        chunk_overlap: 청크 간 오버랩 (문장 개수)
    """
    # 데이터 로드
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # 청커 초기화
    chunker = DataChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 문서 청킹
    chunks = chunker.chunk_documents(documents)
    
    # 청크 저장
    chunker.save_chunks(chunks, output_path)
    
    return chunks


if __name__ == "__main__":
    # 예제 실행
    process_korean_history_data(
        raw_data_path="../data/rag_output.json",
        output_path="../data/processed2/chunks.pkl",
        chunk_size=512,
        chunk_overlap=1  # 문장 1개 오버랩
    )
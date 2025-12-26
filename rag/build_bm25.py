"""
chunks.pkl에서 BM25 인덱스 구축

Dense와 동일한 청크 사용!
"""
import pickle
import sys
import os
from loguru import logger
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from retrieval_bm25 import BM25Retrieval


def load_chunks(chunk_file: str):
    """청크 파일 로드"""
    logger.info(f"청크 파일 로드: {chunk_file}")
    
    with open(chunk_file, 'rb') as f:
        chunks = pickle.load(f)
    
    logger.info(f"총 {len(chunks)}개 청크 로드 완료")
    return chunks


def build_bm25_index(chunks: list, output_path: str):
    """BM25 인덱스 구축"""
    # 문서 및 메타데이터 추출
    documents = []
    metadata = []
    
    for chunk in chunks:
        documents.append(chunk['text'])
        
        # 메타데이터 추출
        meta = {}
        if 'metadata' in chunk:
            meta = chunk['metadata']
        elif 'title' in chunk:
            meta['main_title'] = chunk['title']
        
        metadata.append(meta)
    
    logger.info(f"문서 개수: {len(documents)}")
    
    # BM25 인덱스 구축
    logger.info("BM25 인덱스 구축 시작...")
    retriever = BM25Retrieval()
    retriever.build_index(documents, metadata)
    
    # 저장
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    retriever.save(output_path)
    logger.info(f"BM25 인덱스 저장 완료: {output_path}")
    
    return retriever


def test_search(retriever: BM25Retrieval):
    """검색 테스트"""
    test_queries = [
        "경덕왕",
        "사벌주 상주",
        "가야 건국",
        "연개소문 삼교",
        "구지봉 구간"
    ]
    
    logger.info("\n" + "="*80)
    logger.info("검색 테스트")
    logger.info("="*80)
    
    for query in test_queries:
        logger.info(f"\n쿼리: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            title = result['metadata'].get('main_title', '제목없음')
            score = result['score']
            text = result['text'][:80]
            
            logger.info(f"{i}. [{title}] (score: {score:.3f})")
            logger.info(f"   {text}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BM25 인덱스 구축 (chunks.pkl)")
    parser.add_argument(
        '--chunk-file',
        type=str,
        default='../data/chunks/chunks.pkl',
        help='청크 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/bm25_index.pkl',
        help='BM25 인덱스 저장 경로'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='검색 테스트 실행'
    )
    
    args = parser.parse_args()
    
    # 청크 로드
    chunks = load_chunks(args.chunk_file)
    
    # BM25 인덱스 구축
    retriever = build_bm25_index(chunks, args.output)
    
    # 테스트
    if args.test:
        test_search(retriever)
    
    logger.info("\n완료!")
    logger.info(f"BM25 인덱스: {args.output}")
    logger.info(f"문서 개수: {len(retriever.corpus)}")


if __name__ == "__main__":
    main()
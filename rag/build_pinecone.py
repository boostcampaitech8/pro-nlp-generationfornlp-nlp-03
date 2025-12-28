"""
원본 JSON으로 Pinecone 구축 (청킹 없이) -> 근데 청킹 필요 !!!!!!!!! 
각 section을 별도 벡터로 저장
"""
import os
import json
import time
from tqdm import tqdm
from loguru import logger

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    logger.error("pinecone not installed. Run: pip install pinecone")
    raise

try:
    from openai import OpenAI
except ImportError:
    logger.error("openai not installed. Run: pip install openai")
    raise

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def build_pinecone_from_json(
    json_path="data/raw/rag_output.json",
    index_name="korean-history",
    model_name="solar-embedding-1-large-passage",
    batch_size=100
):
    """
    원본 JSON으로 Pinecone 구축
    """
    logger.info("=" * 80)
    logger.info("Building Pinecone Index from Original JSON")
    logger.info("=" * 80)
    
    # API 키
    solar_api_key = os.getenv("UPSTAGE_API_KEY")
    if not solar_api_key:
        raise ValueError("UPSTAGE_API_KEY not found")
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found")
    
    logger.info(f"Solar API key: {solar_api_key[:10]}...")
    logger.info(f"Pinecone API key: {pinecone_api_key[:10]}...")
    
    # Solar API
    solar_client = OpenAI(
        api_key=solar_api_key,
        base_url="https://api.upstage.ai/v1/solar"
    )
    
    # Pinecone 초기화
    pc = Pinecone(api_key=pinecone_api_key)
    
    # JSON 로드
    logger.info(f"Loading JSON from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # 섹션별로 분리
    sections = []
    for doc_idx, doc in enumerate(documents):
        title = doc.get('title', '')
        sub_title = doc.get('sub_title', '')
        
        for section in doc.get('content', []):
            section_title = section.get('section_title', '')
            section_text = section.get('section_text', '')
            
            if section_text.strip():
                sections.append({
                    'doc_idx': doc_idx,
                    'title': title,
                    'sub_title': sub_title,
                    'section_title': section_title,
                    'text': section_text
                })
    
    logger.info(f"Total sections: {len(sections)}")
    
    # Solar 차원 확인
    test_response = solar_client.embeddings.create(
        input=["test"],
        model=model_name
    )
    vector_dim = len(test_response.data[0].embedding)
    logger.info(f"Vector dimension: {vector_dim}")
    
    # Pinecone 인덱스 생성
    if index_name in pc.list_indexes().names():
        logger.info(f"Deleting existing index: {index_name}")
        pc.delete_index(index_name)
    
    logger.info(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=vector_dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
    # 인덱스 준비 대기
    logger.info("Waiting for index...")
    time.sleep(10)
    
    index = pc.Index(index_name)
    logger.info("✅ Index ready")
    
    # 임베딩 생성 및 업로드
    logger.info("Generating embeddings and uploading...")
    
    # 텍스트 추출
    texts = [s['text'] for s in sections]
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing", total=total_batches):
        batch_texts = texts[i:i + batch_size]
        batch_sections = sections[i:i + batch_size]
        
        try:
            # Solar API로 임베딩 생성
            response = solar_client.embeddings.create(
                input=batch_texts,
                model=model_name
            )
            batch_embeddings = [data.embedding for data in response.data]
            
            # Pinecone 업로드
            vectors = []
            for j, (embedding, section) in enumerate(zip(batch_embeddings, batch_sections)):
                vector_id = f"section_{i + j}"
                
                # 메타데이터 - ID만 저장! (한글 인코딩 문제 회피)
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "doc_idx": section['doc_idx'],
                        "section_idx": i + j
                    }
                })
            
            # 업로드
            index.upsert(vectors=vectors, namespace="")
            
            # Rate limit
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error at batch {i//batch_size}: {e}")
            time.sleep(5)
            continue
    
    # 섹션 정보를 별도 파일로 저장 (한글 포함)
    sections_file = json_path.replace('.json', '_sections.json')
    logger.info(f"Saving sections metadata to {sections_file}")
    with open(sections_file, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    
    # 통계
    stats = index.describe_index_stats()
    
    logger.info("=" * 80)
    logger.info("Pinecone index built successfully!")
    logger.info(f"Index: {index_name}")
    logger.info(f"Total vectors: {stats.total_vector_count}")
    logger.info(f"Vector dimension: {vector_dim}")
    logger.info(f"Sections metadata: {sections_file}")
    logger.info("=" * 80)
    
    return index_name, sections_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json-path',
        type=str,
        default='data/raw/rag_output.json',
        help='Path to rag_output.json'
    )
    parser.add_argument(
        '--index-name',
        type=str,
        default='korean-history',
        help='Pinecone index name'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='solar-embedding-1-large-passage',
        help='Solar model'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size'
    )
    
    args = parser.parse_args()
    
    build_pinecone_from_json(
        json_path=args.json_path,
        index_name=args.index_name,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
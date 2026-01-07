"""
upload_to_pinecone.py - namespace로 주제별 분리
"""
import pickle
import os
import json
from pinecone import Pinecone, ServerlessSpec
from langchain_upstage import UpstageEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def upload_chunks_to_pinecone_with_namespace(
    file_key_json: str = "data/file_key.json",
    index_name: str = "knowledge",
    dimension: int = 4096,
    batch_size: int = 100
):
    """
    주제별 pkl 파일을 namespace로 분리하여 업로드
    """
    
    # 1. file_key.json 로드
    with open(file_key_json, 'r', encoding='utf-8') as f:
        file_map = json.load(f)
    
    # 2. 주제 매핑 (파일명/경로 → namespace)
    def get_subject_from_path(file_path: str) -> str:
        """파일 경로에서 주제 추출"""
        file_path_lower = file_path.lower()
        
        # 우선순위대로 매칭
        if 'korea_history' in file_path_lower or 'history' in file_path:
            return "korean-history"
        elif 'economy' in file_path:
            return "economics"
        elif 'philosophy' in file_path:
            return "philosophy"
        elif 'science' in file_path:
            return "science-tech"
        elif 'social' in file_path:
            return "social-science"
        elif 'humanity' in file_path or 'social' in file_path:
            return "humanities"
        elif 'literature' in file_path_lower:
            return "literature"
        else:
            return "etc"
    
    # 3. Pinecone 초기화
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 인덱스 생성 (없으면)
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    index = pc.Index(index_name)
    
    # 4. Embeddings 초기화
    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="solar-embedding-1-large"
    )
    
    # 5. 각 pkl 파일을 namespace로 업로드
    total_uploaded = 0
    namespace_stats = {}
    
    for source_file, pkl_file in file_map.items():
        # 파일명에서 주제 추출
        subject = get_subject_from_path(source_file)
        
        # Chunks 로드
        print(f"\n{'='*60}")
        print(f"Source: {source_file}")
        print(f"PKL: {pkl_file}")
        print(f"Namespace: {subject}")
        print(f"{'='*60}")
        
        if not os.path.exists(pkl_file):
            print(f"⚠️ File not found: {pkl_file}, skipping...")
            continue
        
        with open(pkl_file, 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"Loaded {len(chunks)} chunks")
        
        # 배치 업로드
        uploaded_count = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc=f"Uploading {subject}"):
            batch = chunks[i:i+batch_size]
            
            vectors = []
            for chunk in batch:
                # 고유 ID 생성
                chunk_id = f"{chunk.get('doc_id', 0)}_{chunk['chunk_id']}"
                text = chunk['text']
                
                # 임베딩 생성
                try:
                    embedding = embeddings.embed_query(text)
                except Exception as e:
                    print(f"⚠️ Embedding error for chunk {chunk_id}: {e}")
                    continue
                
                # 메타데이터 준비
                metadata = {
                    'text': text,
                    'subject': subject,
                    'title': chunk.get('title', ''),
                    'doc_id': chunk.get('doc_id', 0),
                    'chunk_id': chunk['chunk_id'],
                    'sentence_count': chunk.get('sentence_count', 0),
                    'token_count': chunk.get('token_count', 0),
                    'source_file': source_file
                }
                
                # 원본 메타데이터 병합
                if 'metadata' in chunk:
                    for k, v in chunk['metadata'].items():
                        if k not in metadata:
                            # Pinecone 메타데이터는 문자열, 숫자, boolean만 가능
                            if isinstance(v, (str, int, float, bool)):
                                metadata[k] = v
                            elif v is None:
                                metadata[k] = ""
                            else:
                                metadata[k] = str(v)
                
                vectors.append({
                    'id': chunk_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            if vectors:
                # namespace로 업로드
                index.upsert(vectors=vectors, namespace=subject)
                uploaded_count += len(vectors)
        
        total_uploaded += uploaded_count
        namespace_stats[subject] = namespace_stats.get(subject, 0) + uploaded_count
        print(f"✓ Uploaded {uploaded_count} chunks to namespace '{subject}'")
    
    # 6. 통계 출력
    print(f"\n{'='*60}")
    print(f"Upload Complete!")
    print(f"{'='*60}")
    print(f"Index name: {index_name}")
    print(f"Total uploaded: {total_uploaded} chunks")
    
    print(f"\nNamespace breakdown (from upload):")
    for ns, count in sorted(namespace_stats.items()):
        print(f"  {ns}: {count} chunks")
    
    print(f"\nPinecone Index stats:")
    stats = index.describe_index_stats()
    print(f"Total vectors: {stats.get('total_vector_count', 0)}")
    
    if 'namespaces' in stats:
        print(f"\nNamespace breakdown (from Pinecone):")
        for ns, info in sorted(stats['namespaces'].items()):
            print(f"  {ns}: {info.get('vector_count', 0)} vectors")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-key', type=str, default='data/file_key.json',
                       help='Path to file_key.json')
    parser.add_argument('--index-name', type=str, default='knowledge',
                       help='Pinecone index name')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for upload')
    
    args = parser.parse_args()
    
    upload_chunks_to_pinecone_with_namespace(
        file_key_json=args.file_key,
        index_name=args.index_name,
        batch_size=args.batch_size
    )
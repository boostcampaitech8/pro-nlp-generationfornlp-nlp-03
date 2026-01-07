import time
import pickle
import os
import json
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def upload_to_chroma_with_embedding(
    file_key_json: str = "data/file_key.json",
    persist_directory: str = "../DB/chroma_db",
    batch_size: int = 5 # 에러 발생 시 추적이 쉽도록 배치를 작게 조정
):
    with open(file_key_json, 'r', encoding='utf-8') as f:
        file_map = json.load(f)
    
    embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")

    def get_subject_from_path(file_path: str) -> str:
        path = file_path.lower()
        if 'science' in path: return "science-tech"
        elif 'social' in path: return "social-science"
        elif 'human' in path: return "humanities"
        elif 'literature' in path: return "literature"
        else: return "etc"

    total_count = 0

    for source_file, pkl_file in file_map.items():
        subject = get_subject_from_path(source_file)
        if not os.path.exists(pkl_file): continue
        
        print(f"\n[{subject}] 처리 중: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            chunks = pickle.load(f)
        
        if "science_01_chunks" in pkl_file:
            print(f">>> {pkl_file}의 11320번 인덱스부터 재시작합니다.")
            chunks = chunks[11320:]

        vector_db = Chroma(
            collection_name=subject,
            persist_directory=persist_directory,
            embedding_function=embeddings_model
        )

        batch_docs, batch_ids = [], []

        # 에러 발생 지점부터 재시작하고 싶다면, chunks[시작인덱스:] 로 슬라이싱하세요.
        # 예: science_01에서 11320번에서 멈췄다면 chunks[11320:]
        for chunk in tqdm(chunks, desc=f"Embedding {subject}"):
            text = chunk['text']
            
            # [핵심 수정] 400 에러 방지: 텍스트가 너무 길면 강제로 자름 (약 3500토큰 내외 안전권)
            # 1글자를 대략 1~2토큰으로 계산했을 때 안전하게 6000자 정도로 제한
            if len(text) > 6000:
                text = text[:6000] 

            doc = Document(
                page_content=text,
                metadata={
                    'subject': subject,
                    'title': chunk.get('title', ''),
                    'doc_id': chunk.get('doc_id', 0),
                    'chunk_id': chunk.get('chunk_id', 0)
                }
            )
            batch_docs.append(doc)
            batch_ids.append(f"{chunk.get('doc_id', 0)}_{chunk.get('chunk_id', 0)}")

            if len(batch_docs) >= batch_size:
                try:
                    vector_db.add_documents(documents=batch_docs, ids=batch_ids)
                    total_count += len(batch_docs)
                    batch_docs, batch_ids = [], []
                    time.sleep(0.5) # 속도 제한 방지용 짧은 휴식
                except Exception as e:
                    if "429" in str(e):
                        print("\n⚠️ Rate Limit! 20초 대기...")
                        time.sleep(20)
                    elif "400" in str(e):
                        print(f"\n❌ 토큰 초과 데이터 발견 (ID: {batch_ids}), 건너뜁니다.")
                        batch_docs, batch_ids = [], []
                    else:
                        print(f"\n❌ 알 수 없는 에러: {e}")
                        break

        if batch_docs:
            vector_db.add_documents(documents=batch_docs, ids=batch_ids)
            total_count += len(batch_docs)

    print(f"\n✅ 완료! 총 {total_count}개 청크 저장됨.")

if __name__ == "__main__":
    upload_to_chroma_with_embedding()
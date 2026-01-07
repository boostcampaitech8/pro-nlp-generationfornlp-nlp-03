import pickle
import os
import json
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

# 1. BGE-M3 로컬 모델 설정 (V100 GPU 활용)
print("--- 로컬 BGE-M3 모델을 GPU(CUDA)에 로드 중... ---")
model_name = "BAAI/bge-m3"
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'}, 
    encode_kwargs={'normalize_embeddings': True}
)

def get_subject_from_path(file_path: str) -> str:
    """파일 경로 또는 파일명에 따른 주제 매핑"""
    path = file_path.lower()
    if 'history' in path: return "korean-history"
    elif 'economy' in path: return "economics"
    elif 'philosophy' in path: return "philosophy"
    elif 'science' in path: return "science-tech"
    elif 'social' in path: return "social-science"
    elif 'human' in path or 'humanity' in path: return "humanities"
    elif 'literature' in path: return "literature"
    else: return "etc"

def upload_all_to_faiss_gpu(
    file_key_json: str = "data/file_key2.json",
    base_persist_directory: str = "./FAISS/faiss_db"
):
    # JSON 파일 로드
    if not os.path.exists(file_key_json):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {file_key_json}")
        return

    with open(file_key_json, 'r', encoding='utf-8') as f:
        file_map = json.load(f)

    if not os.path.exists(base_persist_directory):
        os.makedirs(base_persist_directory)

    for source_json, pkl_file in file_map.items():
        # 파일명을 추출해서 ID 생성에 사용 (예: economy_01_chunks)
        file_tag = os.path.basename(pkl_file).replace(".pkl", "")
        
        subject = get_subject_from_path(source_json)
        subject_path = os.path.join(base_persist_directory, subject)
        
        if not os.path.exists(pkl_file):
            print(f"⚠️ PKL 파일을 찾을 수 없습니다: {pkl_file}, 건너뜁니다.")
            continue

        print(f"\n[{subject}] 처리 중: {pkl_file}")
        
        # 2. Chunks 로드
        with open(pkl_file, 'rb') as f:
            chunks = pickle.load(f)

        # 3. 기존 인덱스 로드 확인
        if os.path.exists(os.path.join(subject_path, "index.faiss")):
            vector_db = FAISS.load_local(subject_path, embeddings_model, allow_dangerous_deserialization=True)
            print(f"기존 {subject} 인덱스에 데이터를 추가합니다.")
        else:
            vector_db = None
            print(f"새로운 {subject} 인덱스를 생성합니다.")

        # 4. Document 객체 생성 (Preprocessing)
        docs = []
        ids = []
        for chunk in tqdm(chunks, desc=f"Preprocessing {subject}", leave=False):
            text = chunk['text'][:6000] 
            doc = Document(
                page_content=text,
                metadata={
                    'subject': subject,
                    'file': file_tag,
                    'title': chunk.get('title', ''),
                    'doc_id': chunk.get('doc_id', 0),
                    'chunk_id': chunk.get('chunk_id', 0)
                }
            )
            docs.append(doc)
            
            # [핵심 수정] ID에 file_tag를 추가하여 중복 방지
            unique_id = f"{file_tag}_{chunk.get('doc_id', 0)}_{chunk.get('chunk_id', 0)}"
            ids.append(unique_id)

        # 5. FAISS 업로드 (GPU 연산 과정을 tqdm으로 시각화)
        batch_size = 1000
        with tqdm(total=len(docs), desc=f"🚀 Embedding {subject} (GPU)") as pbar:
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]
                
                if vector_db is None:
                    vector_db = FAISS.from_documents(batch_docs, embeddings_model, ids=batch_ids)
                else:
                    vector_db.add_documents(batch_docs, ids=batch_ids)
                
                pbar.update(len(batch_docs))

        # 6. 최종 저장
        if vector_db:
            vector_db.save_local(subject_path)
            print(f"✅ {subject} 저장 완료: {subject_path}")

if __name__ == "__main__":
    upload_all_to_faiss_gpu()
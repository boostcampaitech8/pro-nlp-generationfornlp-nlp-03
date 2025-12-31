import json
import pickle
from pathlib import Path

def merge_pkl_files(file_paths: list, output_path: str):
    """
    개별 pkl 파일을 결합

    Args:
        file_paths: 입력 파일 경로
        output_path: 출력 경로
    """
    all_chunks = []
    
    for path in file_paths:
        if Path(path).exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                all_chunks.extend(data)
                print(f"Loaded {len(data)} chunks from {path}")
        else:
            print(f"Warning: {path} not found.")

    for i, chunk in enumerate(all_chunks):
        chunk['chunk_id'] = i

    with open(output_path, 'wb') as f:
        pickle.dump(all_chunks, f)
    
    print(f"--- Successfully saved {len(all_chunks)} chunks to {output_path} ---")

if __name__ == "__main__":
    with open('data/file_key.json', 'r', encoding='utf-8') as f:
        implement_dict = json.load(f)
        
    file_paths = [path for _, path in implement_dict.items()]
    merge_pkl_files(file_paths, 'chunks.pkl')
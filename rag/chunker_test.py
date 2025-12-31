from chunker import DataChunker

# DataChunker 인스턴스 생성
chunker = DataChunker(
    chunk_size=512,
    chunk_overlap=1,
    tokenizer_name="monologg/kobert"
)

chunks = chunker.load_chunks('./data/processed/science_chunks.pkl')

def print_info(idx):
    print(f"총 청크 수: {len(chunks)}")
    print(f"\n첫 번째 청크:")
    print(f"텍스트: {chunks[idx]['text']}")
    print(f"문장 수: {chunks[idx]['sentence_count']}")
    print(f"토큰 수: {chunks[idx]['token_count']}")
    print(f"문서 ID: {chunks[idx]['doc_id']}")
    print(f"청크 ID: {chunks[idx]['chunk_id']}")
    print(f"제목: {chunks[idx]['title']}")

# 0, 1, 2, etc. 출력 가능
print_info(idx=1)
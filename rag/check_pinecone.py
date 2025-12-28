"""
Pinecone 인덱스 내용 확인
"""
import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Pinecone 연결
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("korean-history")

# 인덱스 통계
stats = index.describe_index_stats()
print("=" * 80)
print("Pinecone Index Stats")
print("=" * 80)
print(f"Total vectors: {stats.total_vector_count}")
print(f"Dimension: {stats.dimension}")
print(f"Namespaces: {stats.namespaces}")
print()

# 샘플 벡터 가져오기 (fetch by ID)
print("=" * 80)
print("Sample Vectors")
print("=" * 80)

# ID로 직접 조회
sample_ids = [f"section_{i}" for i in range(0, 100, 10)]

for vector_id in sample_ids[:5]:
    try:
        result = index.fetch(ids=[vector_id])
        if result and 'vectors' in result and vector_id in result['vectors']:
            vector_data = result['vectors'][vector_id]
            print(f"\nID: {vector_id}")
            print(f"Metadata: {vector_data.get('metadata', {})}")
            print(f"Vector dimension: {len(vector_data.get('values', []))}")
    except Exception as e:
        print(f"Error fetching {vector_id}: {e}")

# sections.json 매핑 확인
print("\n" + "=" * 80)
print("Sections Metadata Mapping")
print("=" * 80)

with open('../data/rag_output_sections.json', 'r', encoding='utf-8') as f:
    sections = json.load(f)

print(f"Total sections in JSON: {len(sections)}")
print("\nFirst 3 sections:")
for i in range(min(3, len(sections))):
    section = sections[i]
    print(f"\nSection {i}:")
    print(f"  Title: {section['title']}")
    print(f"  Sub-title: {section['sub_title']}")
    print(f"  Section title: {section['section_title']}")
    print(f"  Text (first 100 chars): {section['text'][:100]}...")
    print(f"  Doc index: {section['doc_idx']}")

# 특정 검색어로 테스트
print("\n" + "=" * 80)
print("Test Search: '가락국'")
print("=" * 80)

from openai import OpenAI

solar_api_key = os.getenv("UPSTAGE_API_KEY")
solar_client = OpenAI(
    api_key=solar_api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

# 쿼리 임베딩
query = "가락국 나라"
response = solar_client.embeddings.create(
    input=[query],
    model="solar-embedding-1-large-query"
)
query_emb = response.data[0].embedding

# 검색
results = index.query(
    vector=query_emb,
    top_k=5,
    include_metadata=True,
    namespace=""
)

print(f"\nQuery: '{query}'")
print(f"Results: {len(results.matches)}")

for i, match in enumerate(results.matches, 1):
    section_idx = match.metadata.get("section_idx", 0)
    if section_idx < len(sections):
        section = sections[section_idx]
        print(f"\n{i}. Score: {match.score:.4f}")
        print(f"   Title: {section['title']}")
        print(f"   Text: {section['text'][:100]}...")
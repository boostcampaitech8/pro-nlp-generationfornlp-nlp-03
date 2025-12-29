"""
prepare_rag_enhanced_data.py
RAG로 수집한 facts를 데이터에 추가
Pinecone 버전 - 보기별 검색 지원
"""
import os
import sys
import pandas as pd
import json
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rag.rag_collector import RAGCollector


def prepare_rag_enhanced_data(
    input_csv: str,
    output_csv: str,
    rag_config: dict,
    use_choice_search: bool = True,
    use_item_search: bool = True,
    max_fact_length: int = 500  # 추가!
):
    """
    RAG로 external_facts 추가
    
    Args:
        max_fact_length: 각 fact의 최대 길이 (문장 단위)
    """
    # 데이터 로드
    df = pd.read_csv(input_csv)
    print(f"원본 데이터: {len(df)}개")
    
    # 한국사만 필터링 (옵션)
    if 'topic' in df.columns:
        df_korean = df[df['topic'] == '한국사'].copy()
        print(f"한국사 필터링: {len(df_korean)}개")
        df = df_korean
    
    # RAG 초기화
    print("RAG Collector 초기화 중...")
    collector = RAGCollector(**rag_config)
    
    # 각 문제에 대해 facts 수집
    enhanced_data = []
    
    print(f"\nFacts 수집 중...")
    print(f"  선택지별 검색: {'ON' if use_choice_search else 'OFF'}")
    print(f"  보기별 검색: {'ON' if use_item_search else 'OFF'}")
    print()
    
    # 통계
    stats = {
        "reference_question": 0,
        "with_items": 0,
        "with_choices": 0,
        "paragraph_question": 0
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            # 선택지 파싱
            choices = row['choices']
            if isinstance(choices, str):
                import ast
                try:
                    choices = ast.literal_eval(choices)
                except:
                    choices = []
            
            # Facts 수집
            rag_result = collector.collect_facts(
                paragraph=row['paragraph'],
                question=row['question'],
                choices=choices if use_choice_search else None,
                top_k=5,
                use_choice_search=use_choice_search,
                use_item_search=use_item_search,
                max_fact_length=max_fact_length 
            )
            
            # 통계 수집
            search_method = rag_result.get('search_method', 'paragraph_question')
            stats[search_method] = stats.get(search_method, 0) + 1
            
            # 데이터에 추가
            enhanced_row = row.to_dict()
            enhanced_row['external_facts'] = json.dumps(rag_result, ensure_ascii=False)
            enhanced_data.append(enhanced_row)
            
            # 처음 3개는 상세 출력
            if idx < 3:
                print(f"\n{'='*80}")
                print(f"샘플 {idx+1} (ID: {row.get('id', idx)})")
                print(f"{'='*80}")
                print(f"질문: {row['question'][:80]}...")
                print(f"검색 방식: {search_method}")
                print(f"참조 문제: {rag_result.get('is_reference_question', False)}")
                print(f"대상: {rag_result.get('target', 'N/A')}")
                print(f"Facts 개수: {len(rag_result.get('facts', []))}")
                
                if rag_result.get('paragraph_items'):
                    print(f"보기 개수: {len(rag_result['paragraph_items'])}")
                    for label, text in rag_result['paragraph_items'][:2]:
                        print(f"  ({label}) {text[:60]}...")
                
                if rag_result.get('facts'):
                    print(f"첫 번째 fact: {rag_result['facts'][0][:100]}...")
                print('='*80)
            
        except Exception as e:
            print(f"\n⚠️  Error at index {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # 에러 발생 시 빈 facts
            enhanced_row = row.to_dict()
            enhanced_row['external_facts'] = json.dumps(
                {
                    "target": "", 
                    "facts": [], 
                    "sources": [], 
                    "search_method": "error",
                    "error": str(e)
                }, 
                ensure_ascii=False
            )
            enhanced_data.append(enhanced_row)
    
    # 저장
    enhanced_df = pd.DataFrame(enhanced_data)
    enhanced_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"저장 완료: {output_csv}")
    print(f"총 {len(enhanced_df)}개 데이터")
    print(f"\n검색 방식 통계:")
    for method, count in stats.items():
        if count > 0:
            print(f"  {method}: {count}개 ({count/len(enhanced_df)*100:.1f}%)")
    print('='*80)
    
    return enhanced_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/train_topic_sub-2.csv',
                       help='Input CSV file')
    parser.add_argument('--output', type=str, default='../data/train_with_rag.csv',
                       help='Output CSV file')
    parser.add_argument('--index-name', type=str, default='korean-history',
                       help='Pinecone index name')
    parser.add_argument('--sections-file', type=str, default='../data/rag_output_sections.json',
                       help='Sections metadata JSON file')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Upstage API key')
    parser.add_argument('--pinecone-api-key', type=str, default=None,
                       help='Pinecone API key')
    parser.add_argument('--use-choice-search', action='store_true',
                       help='선택지별 검색 사용 (더 정확하지만 느림)')
    parser.add_argument('--use-item-search', action='store_true',
                       help='지문 보기별 검색 사용 (권장)')
    parser.add_argument('--max-fact-length', type=int, default=500,
                       help='각 fact의 최대 길이 (문장 단위, 기본: 500자)')
    
    args = parser.parse_args()
    
    # API 키 확인
    api_key = args.api_key or os.getenv("UPSTAGE_API_KEY")
    pinecone_api_key = args.pinecone_api_key or os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        print("Error: UPSTAGE_API_KEY not found!")
        print("Set it via:")
        print("  1. export UPSTAGE_API_KEY='your-key'")
        print("  2. --api-key 'your-key'")
        sys.exit(1)
    
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY not found!")
        print("Set it via:")
        print("  1. export PINECONE_API_KEY='your-key'")
        print("  2. --pinecone-api-key 'your-key'")
        sys.exit(1)
    
    # 실행
    prepare_rag_enhanced_data(
        input_csv=args.input,
        output_csv=args.output,
        rag_config={
            "index_name": args.index_name,
            "sections_file": args.sections_file,
            "api_key": api_key,
            "pinecone_api_key": pinecone_api_key
        },
        use_choice_search=args.use_choice_search,
        use_item_search=args.use_item_search,
        max_fact_length=args.max_fact_length
    )
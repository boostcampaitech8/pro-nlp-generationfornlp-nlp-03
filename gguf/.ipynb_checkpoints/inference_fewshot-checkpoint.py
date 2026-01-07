"""
Qwen3-Next-80B-A3B-Thinking 모델을 사용한 Few-shot 객관식 문제 추론 스크립트 (토큰 길이 제한 포함)

설정 방법:
1. prepare_fewshot.py를 실행하여 test_with_fewshot.csv 생성
2. llama-server 실행 (별도 터미널)
3. 이 스크립트 실행

필요 파일:
- test_with_fewshot.csv (prepare_fewshot.py로 생성)
- data/output.csv

추론엔진 실행 (별도 터미널):
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
./llama.cpp/build/bin/llama-server -m ./models/Qwen3-Next-80B-A3B-Thinking-UD-Q3_K_XL.gguf -c 32768 -np 2 -cb -fa on --port 8000 --host 0.0.0.0

"""

import pandas as pd
import asyncio
import ast
import time
import os
import re
from collections import Counter
from openai import AsyncOpenAI
from tqdm import tqdm

# ===== 토큰 설정 =====
# 추론 속도 vs 품질 트레이드오프
# - 빠른 추론: CTX_SIZE=16384, MAX_TOKENS=4000 (5-8시간)
# - 균형잡힌: CTX_SIZE=32768, MAX_TOKENS=6000 (12-18시간) ← 현재 설정
# - 고품질: CTX_SIZE=51000, MAX_TOKENS=12000 (33시간)
CTX_SIZE = 32768          # llama-server -c 값 (균형잡힌 설정)
MAX_TOKENS = 6000         # 출력 최대 토큰
SAFETY_MARGIN = 1024      # 여유분
MAX_INPUT_TOKENS = CTX_SIZE - MAX_TOKENS - SAFETY_MARGIN  # 25744 토큰

# Few-shot 비율: 입력의 60%를 few-shot에 할당 (절반보다 조금 더 많이)
FEWSHOT_RATIO = 0.6

# Loop 설정: seed 다양성을 위한 반복 횟수
NUM_LOOPS = 2  # seed 기반 앙상블을 위한 루프 (temperature=1.0이므로 다양성 확보)

# 간단한 토큰 추정 (1 토큰 ≈ 4 글자)
def estimate_tokens(text):
    """텍스트의 대략적인 토큰 수 추정 (한글/영어 혼합)"""
    return len(text) // 4

def truncate_text(text, max_chars):
    """텍스트를 최대 글자 수로 자르기"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[이하 생략]"


# 데이터 로드
df_test = pd.read_csv('test_with_fewshot_fixed.csv')

# Truncation 경고 로그 저장용
truncation_warnings = []

# OpenAI 클라이언트 설정 (로컬 llama.cpp 서버 사용)
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-no-key-required"
)

# System Prompts
system_msg_0 = """당신은 객관식 문제를 해결하는 학생입니다. 제시문과 질문으로 이루어진 문제가 주어집니다.
문제를 해결하기 위한 사고과정을 step-by-step으로 자세히 명시하면서 추론을 수행해주세요. 정답에 이르는 과정은 매우 논리적으로 작성되어야 합니다.
풀이과정이 끝나면 맨 마지막 줄에는 아래의 형식으로 정답을 기재해주세요.
{"정답": "번호"}"""


system_msg_1 = """당신은 객관식 문제를 해결하는 학생입니다. 아래에 유사한 문제의 예시가 제공됩니다.

## 예시 문제들

{few_shot_examples}

---

## 이제 다음 문제를 풀어주세요

제시문과 질문으로 이루어진 문제가 주어집니다.
위 예시들을 참고하여 문제를 해결하기 위한 사고과정을 step-by-step으로 자세히 명시하면서 추론을 수행해주세요.
정답에 이르는 과정은 매우 논리적으로 작성되어야 합니다.
풀이과정이 끝나면 맨 마지막 줄에는 아래의 형식으로 정답을 기재해주세요.
{{"정답": "번호"}}"""


system_msg_3 = """You are a student tasked with solving multiple-choice questions. A problem consisting of a passage and a question will be provided.
Please perform reasoning by explicitly detailing your thought process step-by-step to solve the problem. The process leading to the correct answer must be written very logically.
When the solution process is finished, please write the correct answer on the very last line in the following format:
{{"정답": "number"}}"""


system_msg_4 = """You are a student tasked with solving multiple-choice questions. The problem consists of a passage and a question.
Write extremely clear and evidence-based reasoning according to the instructions below. You must explicitly reveal any lack of background knowledge or logical leaps in your reasoning process. Also, if there is reasoning based on insufficient explanation or incorrect evidence, you must include a process of pointing this out and correcting it yourself.

Instructions:
1. Problem Analysis:
  - First, define what the question is asking for (the core requirement).
  - If background knowledge is needed, describe what knowledge is required and state whether that knowledge is presented in the current passage/problem.
2. Chain of Thought (CoT):
  - Clearly explain why each choice is correct or incorrect by finding evidence in the passage.
  - Do not just find the correct answer; logically explain why the other choices cannot be the answer (process of elimination).
  - Honestly describe points where there are logical leaps or a lack of knowledge-based evidence.
3. Submission Format Compliance:
  - Write the solution process in prose, avoiding unnecessary repetition.
  - Submit the answer on the very last line in the following format:
{{"정답": "number"}}"""



# =========================
# [NEW] Answer-only variants
# =========================

system_msg_1_1 = """당신은 객관식 문제를 해결하는 학생입니다. 아래에 유사한 문제의 예시가 제공됩니다.

## 예시 문제들

{few_shot_examples}

---

## 이제 다음 문제를 풀어주세요

제시문과 질문으로 이루어진 문제가 주어집니다.
위 예시들을 참고하여 문제를 해결하기 위한 사고과정을 step-by-step으로 추론을 수행해주세요.
정답에 이르는 과정은 매우 논리적으로 작성되어야 합니다.

중요 규칙:
- 문제를 풀기 위해 필요한 추론은 충분히 수행하되, 풀이과정/설명/근거는 절대 출력하지 마세요.
- 출력은 오직 마지막 한 줄만 허용됩니다.
- 마지막 한 줄은 반드시 아래 JSON 형식 그대로여야 합니다. 다른 글자, 공백, 줄바꿈을 추가하지 마세요.

출력 형식(마지막 한 줄):
{{"정답": "번호"}}

주의:
- 번호는 1~5 중 하나입니다.
- 위 형식을 어기면 오답 처리될 수 있습니다.
"""


system_msg_4_1 = """You are a student tasked with solving multiple-choice questions. The problem consists of a passage and a question.

You must fully reason internally using the rules below, but you must NOT output any reasoning.

Internal rules (do not print):
- Identify what the question is asking.
- Check whether required knowledge is provided in the passage.
- Evaluate every choice using evidence from the passage.
- Use elimination to reject unsupported or incorrect choices.
- If evidence is weak, correct your reasoning internally before deciding.

Output rules:
- Output ONLY one line.
- The line must be exactly in the following format:
{{"정답": "number"}}
- number must be one of: 1, 2, 3, 4, 5
- Do not add any other text.
"""


# Few-shot 사용
list_system_msg = [system_msg_1_1, system_msg_4_1]


async def get_inference(system_msg, user_content, seed, max_retries=2):
    """단일 추론 요청 수행 (서버 오류 시 재시도 포함)"""
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model="Qwen3-Next-80B-A3B-Thinking",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=MAX_TOKENS,
                temperature=1.0,
                top_p=1.0,
                seed=seed,
            )
            msg = response.choices[0].message

            # Answer-only 모드: 정답 JSON만 반환 (모델은 내부적으로 추론 수행)
            ret = msg.content
            return ret

        except Exception as e:
            if attempt < max_retries:
                print(f"Inference {seed} Error (attempt {attempt+1}/{max_retries+1}): {e}, retrying...")
                await asyncio.sleep(2)  # 2초 대기 후 재시도
            else:
                print(f"Inference {seed} Error (all {max_retries+1} attempts failed): {e}")
                return "Error"


async def process_row(index, row, seed_base, system_msg_template):
    """단일 행(문제)에 대해 few-shot 추론 수행 (토큰 길이 제한 포함)"""
    problem = ast.literal_eval(row['problems'])

    # Few-shot examples 구성
    few_shot_examples = ""
    few_shot_tokens = 0
    max_fewshot_tokens = int(MAX_INPUT_TOKENS * FEWSHOT_RATIO)  # 입력의 60%를 few-shot에 할당

    for shot_num in [1, 2, 3]:
        shot_key = f'shot{shot_num}'
        if pd.notna(row.get(shot_key)):
            shot_text = row[shot_key]
            shot_tokens = estimate_tokens(shot_text)

            # 토큰 초과 시 자르기
            if few_shot_tokens + shot_tokens > max_fewshot_tokens:
                remaining_tokens = max_fewshot_tokens - few_shot_tokens
                if remaining_tokens > 100:  # 최소 100 토큰은 남겨야 의미 있음
                    shot_text = truncate_text(shot_text, remaining_tokens * 4)
                    few_shot_examples += f"### 예시 {shot_num}\n{shot_text}\n\n"

                    # Truncation 경고 로그 추가
                    truncation_warnings.append({
                        'row_index': index,
                        'type': 'few_shot',
                        'shot_num': shot_num,
                        'original_tokens': shot_tokens,
                        'remaining_tokens': remaining_tokens,
                        'message': f"Shot {shot_num} truncated"
                    })
                    print(f"Warning: Shot {shot_num} truncated for row {index}")
                break

            few_shot_examples += f"### 예시 {shot_num}\n{shot_text}\n\n"
            few_shot_tokens += shot_tokens

    # System message에 few-shot 삽입
    system_msg_with_fewshot = system_msg_template.format(few_shot_examples=few_shot_examples)
    system_tokens = estimate_tokens(system_msg_with_fewshot)

    # User content 구성
    paragraph_text = row['paragraph']
    user_content = f"<제시문>\n{paragraph_text}\n\n"
    if pd.notna(row['question_plus']):
        user_content += f"<보기>\n{row['question_plus']}\n\n"
    user_content += f"<질문>\n{problem['question']}\n"
    for k in range(len(problem['choices'])):
        user_content += f"{k+1}. {problem['choices'][k]}\n"

    user_tokens = estimate_tokens(user_content)

    # 전체 입력 토큰 체크
    total_input_tokens = system_tokens + user_tokens

    if total_input_tokens > MAX_INPUT_TOKENS:
        print(f"Warning: Row {index} exceeds token limit!")
        print(f"  System: {system_tokens}, User: {user_tokens}, Total: {total_input_tokens} > {MAX_INPUT_TOKENS}")

        # User content를 줄이기 (제시문만 자르기)
        max_paragraph_chars = (MAX_INPUT_TOKENS - system_tokens - 500) * 4  # 500 토큰 여유
        original_paragraph_tokens = estimate_tokens(paragraph_text)
        paragraph_text = truncate_text(paragraph_text, max_paragraph_chars)

        # Truncation 경고 로그 추가
        truncation_warnings.append({
            'row_index': index,
            'type': 'paragraph',
            'shot_num': None,
            'original_tokens': original_paragraph_tokens,
            'remaining_tokens': max_paragraph_chars // 4,
            'message': f"Paragraph truncated (System: {system_tokens}, User: {user_tokens})"
        })

        # User content 재구성
        user_content = f"<제시문>\n{paragraph_text}\n\n"
        if pd.notna(row['question_plus']):
            user_content += f"<보기>\n{row['question_plus']}\n\n"
        user_content += f"<질문>\n{problem['question']}\n"
        for k in range(len(problem['choices'])):
            user_content += f"{k+1}. {problem['choices'][k]}\n"

    seed = (seed_base * len(df_test) + index)

    # Few-shot system message 사용
    result = await get_inference(system_msg_with_fewshot, user_content, seed)

    # 수정3: 결과에 정답이 없으면 최대 1회 재시도 (submission과 동일한 regex 사용)
    match = re.search(r'정답.*?["\']\s*(\d)\s*["\']', result)
    if not match or match.group(1) not in ['1', '2', '3', '4', '5']:
        print(f"Warning: Row {index} missing answer format, retrying once...")
        result = await get_inference(system_msg_with_fewshot, user_content, seed + 999999)  # 다른 seed로 재시도

    return index, result


async def main():
    """메인 추론 로직"""
    print("Start Inference with Model: Qwen3-Next-80B-A3B-Thinking (Few-shot)")
    print(f"Token Budget: CTX={CTX_SIZE}, MAX_OUTPUT={MAX_TOKENS}, MAX_INPUT={MAX_INPUT_TOKENS}")
    print(f"Ensemble: {NUM_LOOPS} loops × {len(list_system_msg)} system messages = {NUM_LOOPS * len(list_system_msg)} total inferences per row")
    print(f"System Messages: {len(list_system_msg)} templates")

    inference_count = 0
    for s in range(NUM_LOOPS):
        for msg_idx, system_msg_template in enumerate(list_system_msg):
            print(f"\n==== Processing Loop {s+1}/{NUM_LOOPS}, System Msg {msg_idx+1}/{len(list_system_msg)} (inference #{inference_count+1}/{NUM_LOOPS * len(list_system_msg)}) ====")

            # tqdm progress bar로 진행상황 표시
            for i in tqdm(range(len(df_test)), desc=f"Loop {s+1}-{msg_idx+1}", unit="problem"):
                idx, result = await process_row(i, df_test.iloc[i], s * len(list_system_msg) + msg_idx, system_msg_template)

                # 결과 저장
                df_test.loc[idx, f'resp_fewshot_{inference_count}'] = result

                if i % 5 == 4:
                    df_test.to_csv('TestSet_Fewshot.csv', index=False)

            df_test.to_csv('TestSet_Fewshot.csv', index=False)

            inference_count += 1

    # Truncation 경고 CSV 저장
    if truncation_warnings:
        df_warnings = pd.DataFrame(truncation_warnings)
        df_warnings.to_csv('truncation_warnings.csv', index=False)
        print(f"\n⚠️ Truncation warnings saved to truncation_warnings.csv")
        print(f"   Total truncated rows: {len(df_warnings)}")
        print(f"   - Few-shot truncations: {len(df_warnings[df_warnings['type']=='few_shot'])}")
        print(f"   - Paragraph truncations: {len(df_warnings[df_warnings['type']=='paragraph'])}")
    else:
        print("\n✅ No truncation occurred!")


def create_submission():
    """추론 결과를 앙상블하여 최종 제출 파일 생성"""
    # data 폴더 생성 (없으면)
    os.makedirs('./data', exist_ok=True)

    # 추론 결과 로드
    df_test = pd.read_csv('TestSet_Fewshot.csv')

    # 수정1: output.csv를 test의 id로 생성/재구성
    test_ids = df_test['id'].tolist() if 'id' in df_test.columns else list(range(len(df_test)))

    # 기존 output.csv가 있더라도 test id와 다르면 재구성
    if os.path.exists('./data/output.csv'):
        df_output_old = pd.read_csv('./data/output.csv')
        if not df_output_old['id'].equals(pd.Series(test_ids)):
            print("⚠️ output.csv id mismatch with test. Reconstructing with test ids...")
            df_output = pd.DataFrame({'id': test_ids, 'answer': ['1']*len(test_ids)})
        else:
            df_output = df_output_old
    else:
        print("⚠️ output.csv not found. Creating with test ids...")
        df_output = pd.DataFrame({'id': test_ids, 'answer': ['1']*len(test_ids)})

    # 다양성 체크용
    diversity_stats = []

    # 전체 inference 횟수 계산
    total_inferences = NUM_LOOPS * 2  # list_system_msg는 항상 2개

    for i in range(len(df_test)):
        list_choice = []

        # Few-shot 결과 전부 사용 (NUM_LOOPS × 2)
        for inf_idx in range(total_inferences):
            try:
                resp = df_test.loc[i, f'resp_fewshot_{inf_idx}']

                # 공백 제거하고 "정답"과 숫자만 추출
                # {"정답": "4"}, {" 정답 ": "4 "}, {"정답":"4"} 등 모두 처리
                match = re.search(r'정답.*?["\']\s*(\d)\s*["\']', resp)
                if match:
                    choice = match.group(1)
                    if choice in ['1', '2', '3', '4', '5']:
                        list_choice.append(choice)
            except:
                pass

        # 다양성 체크: 같은 답만 반복되는지 확인
        unique_choices = len(set(list_choice))
        diversity_stats.append({
            'row_index': i,
            'total_responses': len(list_choice),
            'unique_choices': unique_choices,
            'choices': list_choice
        })

        list_choice.sort()
        count_choices = Counter(list_choice)
        top2 = count_choices.most_common(2)

        try:
            df_output.loc[i, 'answer'] = top2[0][0]
        except:
            df_output.loc[i, 'answer'] = '1'

    # 다양성 통계 저장 (data 폴더 안에)
    df_diversity = pd.DataFrame(diversity_stats)
    df_diversity.to_csv('./data/ensemble_diversity_check.csv', index=False)

    # 다양성 요약 출력
    all_same_count = len(df_diversity[df_diversity['unique_choices'] == 1])
    print(f"\n📊 Ensemble Diversity Check:")
    print(f"   Total problems: {len(df_diversity)}")
    print(f"   All same answer (no diversity): {all_same_count} ({all_same_count/len(df_diversity)*100:.1f}%)")
    print(f"   Has diversity: {len(df_diversity) - all_same_count} ({(len(df_diversity)-all_same_count)/len(df_diversity)*100:.1f}%)")
    print(f"   Average unique choices: {df_diversity['unique_choices'].mean():.2f}")
    print(f"   Diversity stats saved to: ./data/ensemble_diversity_check.csv")

    # 최종 제출 파일 저장 (data 폴더 안에)
    df_output.to_csv('./data/submission_fewshot.csv', index=False)
    print(f"\n✅ Submission file created: ./data/submission_fewshot.csv")


if __name__ == "__main__":
    # 추론 실행
    time.sleep(5)
    asyncio.run(main())
    time.sleep(5)

    # 제출 파일 생성
    create_submission()

COT_PROMPT_NO_QUESTION_PLUS = """지문:

{paragraph}

질문:
{question}

선택지:
{choices}

아래 형식으로 답하시오.

[핵심 근거]
- 

[선택지 판단]
- 각 선택지가 지문에 충족되는지 근거를 통해 판단하고 1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.

정답:"""

COT_PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

아래 절차에 따라 문제를 해결하시오.

[조건 해석]
- <보기>에 제시된 조건을 의미 단위로 나누어 나열하시오.

[핵심 근거]
- 


정답:<1~5 중 하나> 또는 <1~4 중 하나"""

COT_SYSTEM_MESSAGE = """당신은 수능형 객관식 문제를 푸는 모델입니다. 문제를 단계적으로 분석하여 정답을 도출하십시오.

반드시 다음을 포함하십시오:
- 지문에서 질문과 관련된 핵심 근거
- 각 선택지의 타당성 판단
- 최종 정답 번호
"""

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1 ~ {choice_count} 중에 하나를 정답으로 고르세요. 
아래 형식으로만 답하세요.

정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1 ~ {choice_count} 중에 하나를 정답으로 고르세요. 
아래 형식으로만 답하시오.

정답:"""

SYSTEM_MESSAGE = "지문을 읽고 질문의 답을 구하세요."
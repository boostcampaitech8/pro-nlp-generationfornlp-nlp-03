COT_PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

지문과 질문을 고려하여 정답을 찾는 과정을 5문장 이내로 설명하고
무조건 1 ~ {choice_count} 중에 하나를 정답으로 고르세요.

아래 형식으로만 답하세요.

과정:

정답:
"""


COT_PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

지문과 <보기>를 통해 질문에 대한 정답을 찾는 과정을 5문장 이내로 설명하고
무조건 1 ~ {choice_count} 중에 하나를 정답으로 고르세요.

아래 형식으로만 답하세요.

과정:

정답:"""

COT_SYSTEM_MESSAGE = "당신은 수능형 객관식 문제를 푸는 모델입니다. 문제를 분석하여 하나의 정답을 도출하십시오."

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


PROMPT_NO_QUESTION_PLUS_CHOICE_NUM  = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

정답은 1 ~ {choice_count} 중 숫자 하나만 출력하세요.
아래 형식으로만 답하세요.

정답:"""

PROMPT_QUESTION_PLUS_CHOICE_NUM = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

정답은 1 ~ {choice_count} 중 숫자 하나만 출력하세요.
아래 형식으로만 답하세요.

정답:"""

SYSTEM_MESSAGE_CHOICE_NUM = "너는 객관식 문제 풀이 모델이다. 반드시 정답 번호(숫자) 1개만 출력한다."
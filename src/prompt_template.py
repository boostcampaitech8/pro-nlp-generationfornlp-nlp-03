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
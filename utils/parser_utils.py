import re

def parse_answer(answer_text: str):
    """
    분석용 정답 파서
    - baseline / Mild CoT / Structured CoT 전부 대응
    - CoT 내부 숫자 오인식 방지
    """

    if not answer_text:
        return None

    text = answer_text.strip()

    # 1️⃣ 최우선: 명시적 '정답:' 마커
    m = re.search(r"정답\s*[:：]\s*([1-5])", text)
    if m:
        return int(m.group(1))

    # 2️⃣ [정답] 블록이 있는 경우 (Structured CoT 대비)
    m = re.search(r"\[정답\][\s\S]*?([1-5])", text)
    if m:
        return int(m.group(1))

    # 3️⃣ 마지막 줄만 확인 (fallback)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        m = re.fullmatch(r"([1-5])", lines[-1])
        if m:
            return int(m.group(1))

    # 4️⃣ 실패
    return None

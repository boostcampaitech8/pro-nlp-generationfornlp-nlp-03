# %%
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import re


model_id = "Qwen/Qwen3-4B-Instruct-2507"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False,
    device_map="auto",
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

# %%
paragraph = "북쪽 구지에서 이상한 소리로 부르는 것이 있었다.…(중략) … 구간(九干)들은 이 말을 따라 모두 기뻐하면서 노래하고 춤을 추었다. 자줏빛줄이 하늘에서 드리워져서 땅에 닿았다. 그 줄이 내려온 곳을 따라가 붉은 보자기에 싸인 금으로 만든 상자를 발견하고 열어보니, 해처럼 둥근 황금알 여섯 개가 있었다. 알여섯이 모두 변하여 어린 아이가 되었다.…(중략) … 가장 큰 알에서 태어난 수로(首露)가 왕위에 올라(가)를/ 을 세웠다.－삼국유사－"
question = "(가) 나라에 대한 설명으로 옳은 것은?"
choices = ['해상교역을 통해 우수한 철을 수출하였다.', '박, 석, 김씨가 교대로 왕위를 계승하였다.', '경당을 설치하여 학문과 무예를 가르쳤다.', '정사암회의를 통해 재상을 선발하였다.']
answer = 1

# %%
from typing_extensions import TypedDict

class AgentState(TypedDict):
    paragraph : str
    question: str
    choices: list
    subject: str
    queries: list[str]
    retrieval: list[str]
    answer: str

# %%
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# %%
subject_query = """
아래 지문과 문제를 보고 과목을 분류하세요.
지문: {paragraph}
문제: {question}
선택지: {choices}

과목 후보: 철학, 한국사, 심리, 문학

답변 (과목명만):"""

def subject_finder(state: AgentState) -> AgentState:
    messages = PromptTemplate.from_template(subject_query)
    chain = messages | llm | StrOutputParser()
    ai_message = chain.invoke({"paragraph": state["paragraph"],
                                "question": state["question"],
                                "choices": state["choices"]})
    subject = ai_message.strip().split()[0] if ai_message.strip() else ""
    return {"subject": subject}

# %%
# hypothesis_query = """
# 당신은 검색 최적화 전문가입니다. 주어진 문제를 분석하여 정답을 찾기 위해 필요한 구체적인 파생 질문들을 생성합니다.

# 과목: {subject}
# 지문: {paragraph}
# 질문: {question}
# 선택지: {choices}
# ###SEARCH_START###와 ###SEARCH_END### 사이에만 파생 질문들을 파이썬 리스트 형식으로 작성하세요.

# 다음 규칙을 따르세요.
# - 위 과목의 지문을 참고하여 선택지 중 질문에 가장 적합한 답변을 만들기 위해 다음의 순서로 짧은 파생 질문을 파이썬 리스트 형식으로 생성하세요.
# - 질문과 관련된 지문의 핵심 키워드를 질문과 연관지은 검색어을 생성하세요.
# - 각 선택지를 질문과 연관지은 검색어만을 생성하세요.
# - 파생 질문은 간결하고 구체적이어야 합니다. 다른 설명이나 부연은 절대 포함하지 마세요.
# ###SEARCH_START###"""

hypothesis_query = """당신은 검색 전문가입니다. 아래의 한국사 문제를 해결하기 위해, 검색 엔진에 입력할 '질문 리스트'를 파이썬 리스트 형식으로 생성하십시오.

### 생성 규칙:
1. **첫 번째 요소**: 질문과 관련된 지문의 핵심 키워드를 질문과 연관지은 질문을 생성하세요.
2. **나머지 요소**: 각 선택지를 질문과 연관지은 질문을 생성하세요.
3. **금지 사항**: "지문의 핵심 키워드", "선택지 1" 같은 서술어는 절대 포함하지 마십시오. 오직 파이썬 리스트만 출력하십시오.
4. **질문 형식**: "~는 무엇인가?", "~의 특징은?"과 같은 완전한 의문문 형태로 작성하십시오.
5. **생성 규칙**: '질문 리스트' 생성이 끝나면 종료하세요. 추가적인 출력을 하지마세요.

### 출력 예시:
["질문 1", "질문 2", "질문 3", ...]

과목: {subject}
지문: {paragraph}
질문: {question}
선택지: {choices}

"""

# HyDE, Step-back Prompting
def mid_hypothesis(state: AgentState) -> AgentState:
    messages = PromptTemplate.from_template(hypothesis_query)
    chain = messages | llm | StrOutputParser()
    ai_message = chain.invoke({"paragraph": state["paragraph"],
                                "question": state["question"],
                                "choices": state["choices"],
                                "subject": state["subject"]})
    print(ai_message)
    pattern = r"###SEARCH_START###(.*?)###SEARCH_END###"
    match = re.search(pattern, ai_message, re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        queries = [line.strip("- ").strip() for line in content.split("\n") if line.strip()]
    else:
        queries = [state["question"]] 

    return {"queries": queries}

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# %%
import json
from langchain_core.documents import Document

# %%
def load_korea_history_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    for item in data:
        title = item.get('title', '')
        sub_title = item.get('sub_title', '')
        url = item.get('url', '')
        content_list = item.get('content', [])
        
        for content in content_list:
            section_title = content.get('section_title', '')
            section_text = content.get('section_text', '')
            
            doc = Document(
                page_content=section_text,
                metadata={
                    'title': title,
                    'sub_title': sub_title,
                    'section_title': section_title,
                    'url': url
                }
            )
            documents.append(doc)
    
    return documents

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[". ", " "],
    length_function=len
)


documents = load_korea_history_json("./data/korea_history.json")

document_list = []
for doc in documents:
    if len(doc.page_content) <= 100:
        document_list.append(doc)
    else:
        splits = text_splitter.split_documents([doc])
        document_list.extend(splits)

# %% [markdown]
# ### 최초 1회만

# %%
# from langchain_chroma import Chroma

# database = Chroma.from_documents(documents=document_list, embedding=embeddings, persist_directory="./chroma")

# %% [markdown]
# ### 이후

# %%
from langchain_chroma import Chroma

database = Chroma(persist_directory="./chroma", embedding_function=embeddings)

# %%
def retrieval(state: AgentState):
    queries = state['queries']
    
    results = []
    for query in queries:
        docs = database.similarity_search(query, k=1)
        if docs:
            results.append(docs[0].page_content)
        else:
            results.append("검색 결과 없음")
            
    return {"retrieval": results}

# %%
generate_query = """
당신은 {subject} 과목 문제 풀이 전문가입니다. 지문, 질문, 참고자료를 바탕으로 선택지 중 가장 적합한 답변을 생성하세요.
지문: {paragraph}
질문: {question}
참고자료: {reference}
선택지: {choices}

답변은 선택지 중 하나만 생성하세요. 다른 설명은 절대 하지 마세요.

답변(선택지 중 하나만 답변):"""

def generate(state: AgentState) -> AgentState:
    messages = PromptTemplate.from_template(generate_query)
    chain = messages | llm | StrOutputParser()
    
    queries = state["queries"]
    retrieval = state["retrieval"]

    reference = ""
    for query, retrieve in zip(queries, retrieval):
        template = f"{query}에 대한 참고자료: {retrieve}"
        reference += template + "\n"

    answer = chain.invoke({"paragraph": state["paragraph"],
                            "question": state["question"],
                            "choices": state["choices"],
                            "subject": state["subject"],
                            "reference": reference})
    answer = answer.strip().split('\n')[0].strip()
    return {"answer": answer}

# %%
from langgraph.graph import START, END

graph_builder.add_node('subject_finder', subject_finder)
graph_builder.add_node('mid_hypothesis', mid_hypothesis)
graph_builder.add_node('retrieval', retrieval)
graph_builder.add_node('generate', generate)

graph_builder.add_edge(START, 'subject_finder')
graph_builder.add_edge('subject_finder', 'mid_hypothesis')
graph_builder.add_edge('mid_hypothesis','retrieval')
graph_builder.add_edge('retrieval', 'generate')
graph_builder.add_edge('generate', END)

# %%
graph = graph_builder.compile()

# %%
# from IPython.display import display, Image

# display(Image(graph.get_graph().draw_mermaid_png()))

# %%
initial_state = {'paragraph': paragraph, 'question': question, 'choices': choices}
graph.invoke(initial_state)

# %%


# %%




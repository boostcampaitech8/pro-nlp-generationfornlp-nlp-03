from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from transformers import pipeline
from typing_extensions import TypedDict
from unsloth import FastLanguageModel

import torch
import json
import yaml
import re
import os
from dotenv import load_dotenv


def get_torch_dtype(dtype_str: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float16)


class AgentState(TypedDict):
    paragraph: str
    question: str
    question_plus: str
    choices: list
    subject: str
    queries: list[str]
    retrieval: list[str]
    retrieval_by_query: dict
    scored_retrieval: list
    answers: int
    shot1: str
    shot2: str
    shot3: str

    iter: int
    topk: int
    threshold: float
    should_retry: bool
    score_topn: int


CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\\n' }}{% endif %}{% endfor %}"


def setup_tokenizer(tokenizer):
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


class Agent:
    def __init__(self, model_id="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"):
        print(f"Loading model: {model_id}")

        # Unsloth로 모델 로딩
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            dtype=get_torch_dtype("float16"),
            load_in_4bit=True,
            max_seq_length=8192,  # 4096 → 8192로 증가
        )
        self.tokenizer = setup_tokenizer(self.tokenizer)
        self.model = FastLanguageModel.for_inference(self.model)

        # Pipeline 생성
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,  # 512 → 1024로 증가
            temperature=0.1,
            do_sample=False,
            device_map="auto",
            return_full_text=False,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        # BGE-M3 임베딩
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ BGE-M3 임베딩 로드 완료")

        # FAISS 디렉토리
        self.faiss_directory = "/data/ephemeral/iseul/rag/FAISS/faiss_db"
        self.databases = {}

        self.graph_builder = StateGraph(AgentState)

        # Prompts
        self.subject_prompt = None
        self.query_prompt = None
        self.generate_prompt = None
        self.score_prompt = None
        self.refine_prompt = None

        self.set_queries(file_path="./PROMPT/PROMPTS.yaml")
        self.graph = None

    def load_faiss_db(self, collection_name):
        """FAISS DB 로드"""
        if collection_name in self.databases:
            return self.databases[collection_name]

        faiss_path = os.path.join(self.faiss_directory, collection_name)
        if not os.path.exists(faiss_path):
            return None

        try:
            db = FAISS.load_local(
                faiss_path, self.embeddings, allow_dangerous_deserialization=True
            )
            self.databases[collection_name] = db
            return db
        except Exception as e:
            print(f"Error loading {collection_name}: {e}")
            return None

    def set_queries(self, file_path="./PROMPT/PROMPTS.yaml"):
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        subject = config["subject"]
        query = config["query"]
        generate = config["generate"]
        score = config.get("score", {})
        refine = config.get("refine_query", {})

        self.subject_prompt = ChatPromptTemplate.from_messages(
            [("system", subject["system"]), ("user", subject["user"])]
        )
        self.query_prompt = ChatPromptTemplate.from_messages(
            [("system", query["system"]), ("user", query["user"])]
        )
        self.generate_prompt = ChatPromptTemplate.from_messages(
            [("system", generate["system"]), ("user", generate["user"])]
        )
        self.score_prompt = ChatPromptTemplate.from_messages(
            [("system", score.get("system", "")), ("user", score.get("user", ""))]
        )
        self.refine_prompt = ChatPromptTemplate.from_messages(
            [("system", refine.get("system", "")), ("user", refine.get("user", ""))]
        )

    def subject_finder(self, state: AgentState) -> AgentState:
        chain = self.subject_prompt | self.llm | StrOutputParser()
        ai_message = chain.invoke(
            {
                "paragraph": state["paragraph"],
                "question": state["question"],
                "choices": state["choices"],
            }
        )

        pattern = r'"subject":\s*"([^"]*)"'
        match = re.search(pattern, ai_message)

        if match:
            subject = match.group(1).strip()
        else:
            subject = ""
        
        return {"subject": subject}

    def generate_query(self, state: AgentState) -> AgentState:
        chain = self.query_prompt | self.llm | StrOutputParser()
        queries = chain.invoke(
            {
                "paragraph": state["paragraph"],
                "question": state["question"],
                "choices": state["choices"],
                "subject": state["subject"],
            }
        )
        pattern = r'"query":\s*\[(.*?)\]'
        match = re.search(pattern, queries, re.DOTALL)

        if match:
            content = match.group(1).strip()
            queries = re.findall(r'"([^"]*)"', content)
        else:
            queries = [state["question"]]
        return {"queries": queries}

    def retrieval(self, state: AgentState):
        queries = state["queries"]
        raw_subject = state.get("subject", "")
        results = []
        retrieval_by_query = {}

        subject_mapping = {
            "한국사": "korean-history",
            "경제": "economics",
            "철학": "philosophy",
            "과학기술": "science-tech",
            "사회과학": "social-science",
            "인문학": "humanities",
            "문학": "literature",
        }

        target_col = subject_mapping.get(raw_subject)
        if not target_col:
            return {"retrieval": [], "retrieval_by_query": {}}

        for query in queries:
            all_candidate_chunks = []
            k_val = state.get("topk", 3)

            db = self.load_faiss_db(target_col)
            if db is None:
                results.append("검색 결과 없음")
                retrieval_by_query[query] = []
                continue

            try:
                docs_and_scores = db.similarity_search_with_score(query, k=k_val)

                for doc, score in docs_and_scores:
                    similarity = 1 / (1 + score) if score >= 0 else 0.99
                    all_candidate_chunks.append(
                        {
                            "chunk": doc.page_content,
                            "sim": similarity,
                            "ref": target_col,
                            "query": query,
                        }
                    )
            except Exception as e:
                print(f"Error searching in {target_col}: {e}")
                continue

            all_candidate_chunks.sort(key=lambda x: x["sim"], reverse=True)
            top_chunks = all_candidate_chunks[:k_val]

            if top_chunks:
                combined_content = "\n\n".join([c["chunk"] for c in top_chunks])
                results.append(combined_content)
                retrieval_by_query[query] = top_chunks
            else:
                results.append("검색 결과 없음")
                retrieval_by_query[query] = []

        return {"retrieval": results, "retrieval_by_query": retrieval_by_query}

    def score_retrieval(self, state: AgentState) -> AgentState:
        chain = self.score_prompt | self.llm | StrOutputParser()
        score_topn = state.get("score_topn", 2)
        scored = []

        by_query = state.get("retrieval_by_query")
        if by_query is None:
            by_query = {}
            for item in state.get("retrieval", []):
                by_query.setdefault(item["query"], []).append(item)
            for q in by_query:
                by_query[q].sort(key=lambda x: x["sim"], reverse=True)

        for q, items in by_query.items():
            top_items = items[:score_topn]
            for item in top_items:
                resp = chain.invoke({"query": item["query"], "chunk": item["chunk"]})

                validity = 0
                try:
                    cleaned = resp.strip()
                    cleaned = re.sub(
                        r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE
                    ).strip()
                    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    if m:
                        cleaned = m.group(0)
                    data = json.loads(cleaned)
                    validity = float(data.get("validity", 0))
                except:
                    validity = 2

                final = 0.7 * (validity / 5) + 0.3 * item["sim"]
                scored.append(
                    {"query": item["query"], "chunk": item["chunk"], "final": final}
                )

        best_scores = {}
        for s in scored:
            q = s["query"]
            best_scores[q] = max(best_scores.get(q, 0), s["final"])
        avg_score = sum(best_scores.values()) / max(len(best_scores), 1)
        should_retry = avg_score < state["threshold"] and state["iter"] < 3

        return {"scored_retrieval": scored, "should_retry": should_retry}

    def refine_search(self, state: AgentState) -> AgentState:
        chain = self.refine_prompt | self.llm | StrOutputParser()
        resp = chain.invoke(
            {
                "subject": state["subject"],
                "paragraph": state["paragraph"],
                "question": state["question"],
                "choices": state["choices"],
                "queries": state["queries"],
            }
        )
        try:
            cleaned = re.sub(
                r"^```json\s*|\s*```$", "", resp.strip(), flags=re.IGNORECASE
            ).strip()
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            data = json.loads(m.group(0)) if m else json.loads(cleaned)
            new_queries = data.get("query", state["queries"])
        except:
            new_queries = state["queries"]

        return {
            "queries": new_queries,
            "iter": state["iter"] + 1,
            "topk": min(state["topk"] + 2, 10),
            "threshold": state["threshold"] - 0.1,
        }

    def generate(self, state: AgentState) -> AgentState:        
        chain = self.generate_prompt | self.llm | StrOutputParser()
        
        if state.get("retrieval"):
            queries = state["queries"]
            retrieval = state["retrieval"]

            reference = ""
            for query, retrieve in zip(queries, retrieval):
                template = f"{query}에 대한 참고자료: {retrieve}"
                reference += template + "\n"
        else:
            reference = "참고자료 없음. 주어진 지문만으로 답변하세요."

        answer = chain.invoke(
            {
                "paragraph": state["paragraph"],
                "question": state["question"],
                "choices": state["choices"],
                "question_plus": state.get("question_plus", ""),
                "subject": state["subject"],
                "reference": reference,
                "shot1": state["shot1"],
                "shot2": state["shot2"],
                "shot3": state["shot3"],
            }
        )

        print(answer)
        pattern = r'"answer":\s*(\d+)'
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            answer = int(match.group(1))
        else:
            print("=" * 80)
            print("Not Answer")
            print("=" * 80)
            answer = 1
        return {"answers": answer}

    def set_graph(self):
        gb = self.graph_builder
        gb.add_node("subject_finder", self.subject_finder)
        gb.add_node("generate_query", self.generate_query)
        gb.add_node("retrieval", self.retrieval)
        gb.add_node("score_retrieval", self.score_retrieval)
        gb.add_node("refine_search", self.refine_search)
        gb.add_node("generate", self.generate)

        gb.add_edge(START, "subject_finder")
        gb.add_edge("subject_finder", "generate_query")
        gb.add_edge("generate_query", "retrieval")
        gb.add_edge("retrieval", "score_retrieval")

        gb.add_conditional_edges(
            "score_retrieval",
            lambda s: "retry" if s["should_retry"] else "generate",
            {"retry": "refine_search", "generate": "generate"},
        )

        gb.add_edge("refine_search", "retrieval")
        gb.add_edge("generate", END)
        self.graph = gb.compile()

    def display(self, save_path=None):
        from IPython.display import display, Image

        png_bytes = self.graph.get_graph().draw_mermaid_png()

        # 화면 출력
        display(Image(png_bytes))

        # 파일 저장
        if save_path:
            with open(save_path, "wb") as f:
                f.write(png_bytes)
            print(f"✓ Graph image saved to {save_path}")

    def __call__(self, state, display=False):
        self.set_graph()
        if display:
            self.display()
        return self.graph.invoke(state)


if __name__ == "__main__":
    load_dotenv()

    import pandas as pd
    import ast
    import pprint
    import time
    from datetime import datetime

    df = pd.read_csv("test_with_fewshot_fixed.csv")

    agent = Agent()
    agent.set_graph()
    agent.display(save_path="agent_graph.png")

    # test 결과 저장
    results = []
    
    # 전체 시작 시간
    total_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"추론 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 {len(df)}개 문제 처리 예정")
    print(f"{'='*80}\n")

    for idx, row in df.iterrows():
        # 문제별 시작 시간
        problem_start_time = time.time()
        
        paragraph = row["paragraph"]
        problems = ast.literal_eval(row["problems"])

        question = problems["question"]
        choices = problems["choices"]

        question_plus = row["question_plus"] if pd.notna(row["question_plus"]) else ""

        shot1 = row["shot1"] if pd.notna(row["shot1"]) else ""
        shot2 = row["shot2"] if pd.notna(row["shot2"]) else ""
        shot3 = row["shot3"] if pd.notna(row["shot3"]) else ""

        state = {
            "paragraph": paragraph,
            "question": question,
            "choices": choices,
            "question_plus": question_plus,
            "shot1": shot1,
            "shot2": shot2,
            "shot3": shot3,
            "iter": 0,
            "topk": 3,
            "threshold": 0.65,
            "score_topn": 2,
        }

        result = agent.graph.invoke(state)

        # 결과에서 답변 추출
        answer = result.get("answers", "")
        
        # 문제별 소요시간 계산
        problem_elapsed = time.time() - problem_start_time
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(df)}] ID: {row['id']}")
        print(f"주제: {result.get('subject', 'N/A')}")
        print(f"예측 답변: {answer}")
        print(f"소요 시간: {problem_elapsed:.2f}초")
        print(f"{'='*80}")

        # 결과 저장
        results.append(
            {
                "id": row["id"],
                "question": question,
                "predicted_answer": answer,
                "choices": choices,
                "subject": result.get("subject", ""),
                "elapsed_time": round(problem_elapsed, 2),
            }
        )

    # 전체 소요시간 계산
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"추론 완료!")
    print(f"{'='*80}")
    print(f"총 처리 문제: {len(results)}개")
    print(f"총 소요 시간: {total_elapsed:.2f}초 ({total_elapsed/60:.2f}분)")
    print(f"평균 소요 시간: {total_elapsed/len(results):.2f}초/문제")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # 결과를 DataFrame으로 변환
    result_df = pd.DataFrame(results)

    # CSV로 저장
    result_df.to_csv("predictions.csv", index=False, encoding="utf-8-sig")
    
    # 통계 출력
    print("\n과목별 통계:")
    print("-" * 80)
    subject_stats = result_df.groupby('subject').agg({
        'id': 'count',
        'elapsed_time': ['mean', 'sum']
    }).round(2)
    subject_stats.columns = ['문제 수', 'RAG 사용', '평균 시간(초)', '총 시간(초)']
    print(subject_stats)
    
    print(f"\n결과 저장 완료: predictions.csv")
import json
import os
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from groq import Groq
from core.retriever import HybridRetriever
from core.cache import SemanticCache
from core.prompt_loader import load_prompt
from config.settings import GROQ_API_KEY, CHAT_MODEL, HF_TOKEN

# 1. Định nghĩa State của Graph
class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    contexts: List[str]
    response: str
    history: List[dict]

class AgenticChatbot:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.retriever = HybridRetriever(HF_TOKEN)
        self.cache = SemanticCache() # Khởi tạo Semantic Cache
        self.workflow = self._create_graph()

    def _create_graph(self):
        workflow = StateGraph(AgentState)

        # Thêm các Node
        workflow.add_node("decompose", self.decompose_query)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("synthesize", self.synthesize_response)

        # Thiết lập các cạnh (Edges)
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "retrieve")
        workflow.add_edge("retrieve", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # Node 1: Query Decomposition
    def decompose_query(self, state: AgentState):
        query = state["query"]
        print(f"--- Đang phân rã câu hỏi: {query} ---")
        
        template = load_prompt("decompose_query", hub_path="hust-rag/hust-decompose-query")
        prompt = template.format(query=query)

        completion = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            content = completion.choices[0].message.content
            data = json.loads(content)
            sub_queries = data.get("sub_queries", [query])
        except:
            sub_queries = [query]

        return {"sub_queries": sub_queries}

    # Node 2: Multi-threaded Retrieval
    def retrieve_context(self, state: AgentState):
        sub_queries = state["sub_queries"]
        all_contexts = []
        
        print(f"--- Đang tìm kiếm cho {len(sub_queries)} truy vấn con ---")
        for sq in sub_queries:
            docs = self.retriever.retrieve(sq, top_k=5, rerank_top_n=3)
            for doc in docs:
                context_str = f"[Nguồn: {doc.metadata.get('source')}]\n{doc.page_content}"
                if context_str not in all_contexts:
                    all_contexts.append(context_str)
        
        return {"contexts": all_contexts}

    # Node 3: Synthesis
    def synthesize_response(self, state: AgentState):
        query = state["query"]
        contexts = "\n\n---\n\n".join(state["contexts"])
        
        print("--- Đang tổng hợp câu trả lời cuối cùng ---")
        
        template = load_prompt("synthesize_response", hub_path="hust-rag/hust-synthesize-response")
        prompt = template.format(contexts=contexts, query=query)

        completion = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return {"response": completion.choices[0].message.content}

    def chat(self, user_input: str, history: List[dict] = None):
        # 1. Kiểm tra Semantic Cache trước
        cached_response = self.cache.get(user_input)
        if cached_response:
            return {
                "response": cached_response,
                "sub_queries": ["(Từ Cache)"],
                "num_sources": 0,
                "from_cache": True
            }

        # 2. Nếu không có cache, chạy luồng Agentic
        initial_state = {
            "query": user_input,
            "sub_queries": [],
            "contexts": [],
            "response": "",
            "history": history or []
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        # 3. Cập nhật kết quả vào Cache cho lần sau
        self.cache.update(user_input, final_state["response"])
        
        return {
            "response": final_state["response"],
            "sub_queries": final_state["sub_queries"],
            "num_sources": len(final_state["contexts"]),
            "from_cache": False
        }

if __name__ == "__main__":
    chatbot = AgenticChatbot()
    test_query = "Tôi bị cảnh cáo học tập mức 2 thì có được đăng ký học vượt không?"
    
    print("\n--- Lần chạy 1 (Chưa có cache) ---")
    result1 = chatbot.chat(test_query)
    print(f"Bot: {result1['response']}")
    
    print("\n--- Lần chạy 2 (Kiểm tra cache với câu hỏi tương tự) ---")
    test_query_2 = "Cảnh cáo học tập mức 2 có được học vượt không?"
    result2 = chatbot.chat(test_query_2)
    print(f"Bot: {result2['response']}")
    print(f"From Cache: {result2.get('from_cache')}")

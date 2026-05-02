import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevance
faithfulness = Faithfulness()
answer_relevance = AnswerRelevance()
from core.chatbot import AgenticChatbot
from config.settings import GEMINI_API_KEY
from google import genai

# 1. Khởi tạo Chatbot
chatbot = AgenticChatbot()

# 2. Tạo tập dữ liệu kiểm thử (Mẫu)
# Trong thực tế, bạn nên chuẩn bị bộ Q&A này thật kỹ lưỡng
test_samples = [
    {
        "question": "Sinh viên năm mấy thì bắt đầu bị xét cảnh báo học tập?",
        "ground_truth": "Việc xét mức cảnh báo học tập được thực hiện sau mỗi học kỳ chính đối với tất cả sinh viên."
    },
    {
        "question": "Mức cảnh báo học tập 3 (mức 3) có hậu quả gì?",
        "ground_truth": "Sinh viên bị cảnh báo học tập mức 3 sẽ bị buộc thôi học."
    },
    {
        "question": "Điều kiện để đăng ký học vượt (trên 24 tín chỉ) là gì?",
        "ground_truth": "Sinh viên không bị cảnh báo học tập có thể đăng ký tối đa 24 TC. Việc học vượt yêu cầu không bị cảnh báo và tuân thủ kế hoạch học tập."
    }
]

def run_evaluation():
    print("--- BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ RAGAS ---")
    
    results_data = []
    
    for sample in test_samples:
        print(f"Đang kiểm tra câu hỏi: {sample['question']}")
        
        # Chạy chatbot để lấy câu trả lời và ngữ cảnh
        # Lưu ý: invoke trực tiếp để lấy contexts cho RAGAS
        initial_state = {
            "query": sample["question"],
            "sub_queries": [],
            "contexts": [],
            "response": "",
            "history": []
        }
        final_state = chatbot.workflow.invoke(initial_state)
        
        results_data.append({
            "question": sample["question"],
            "answer": final_state["response"],
            "contexts": final_state["contexts"],
            "ground_truth": sample["ground_truth"]
        })

    # Chuyển sang format Dataset của HuggingFace (Ragas yêu cầu)
    dataset = Dataset.from_list(results_data)

    # 3. Cấu hình mô hình đánh giá (Gemma-3 qua google-genai)
    # RAGAS hiện tại hỗ trợ OpenAI hoặc LangChain LLMs. 
    # Chúng ta sẽ dùng LangChain Google GenAI wrapper để tích hợp.
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", # Thay bằng gemma-3-27b-it nếu endpoint khả dụng
        google_api_key=GEMINI_API_KEY
    )
    
    print("--- Đang tính toán các chỉ số RAGAS ---")
    # Đánh giá độ trung thực (Faithfulness) và độ liên quan (Answer Relevance)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevance],
        llm=eval_llm
    )

    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ:")
    print(result)
    
    # Lưu kết quả ra CSV
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("\nKết quả chi tiết đã được lưu tại 'evaluation_results.csv'")

if __name__ == "__main__":
    # Cần cài đặt langchain-google-genai cho bước này
    # pip install langchain-google-genai
    try:
        run_evaluation()
    except Exception as e:
        print(f"Lỗi khi đánh giá: {e}")

from core.chatbot import AgenticChatbot
import sys

def main():
    print("--- HUST Academic Regulations Agentic RAG Chatbot ---")
    print("Nhập câu hỏi của bạn. Nhập 'quit' hoặc 'exit' để thoát.")
    
    try:
        chatbot = AgenticChatbot()
    except Exception as e:
        print(f"Lỗi khởi tạo chatbot: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input:
                continue

            result = chatbot.chat(user_input)
            
            print(f"\nSub-queries: {result['sub_queries']}")
            if result.get('from_cache'):
                print("(Kết quả từ Semantic Cache)")
            
            print(f"\nBot: {result['response']}")
            print(f"\n[Thông tin: Lấy từ {result['num_sources']} nguồn ngữ nghĩa]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()

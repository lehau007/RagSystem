from core.chatbot import AgenticChatbot
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="HUST Academic Regulations Agentic RAG")

# Khởi tạo chatbot duy nhất
chatbot = AgenticChatbot()

class ChatRequest(BaseModel):
    user_input: str
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    response: str
    sub_queries: List[str]
    num_sources: int
    from_cache: bool

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = chatbot.chat(request.user_input, history=request.history)
        return ChatResponse(
            response=result["response"],
            sub_queries=result["sub_queries"],
            num_sources=result["num_sources"],
            from_cache=result.get("from_cache", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

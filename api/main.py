from core.chatbot import Chatbot
from config.settings import HF_TOKEN, OSSAPI_KEY
from fastapi import FastAPI
import uvicorn

app = FastAPI() 

chatbot = Chatbot(OSSAPI_KEY, HF_TOKEN)

@app.post("/chat")
def chat(user_input: str):
    return chatbot.chat(user_input)
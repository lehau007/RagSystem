from chatbot import Chatbot
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

app = FastAPI() 

load_dotenv()
hf_token = load_dotenv()
ossapi_key = load_dotenv()
chatbot = Chatbot(ossapi_key, hf_token)

@app.post("/chat")
def chat(user_input: str):
    return chatbot.chat(user_input)

uvicorn.run(app, host="0.0.0.0", port=8000)
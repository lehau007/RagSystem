from core.chatbot import Chatbot
from config.settings import HF_TOKEN, OSSAPI_KEY

if __name__ == "__main__":
    chatbot = Chatbot(OSSAPI_KEY, HF_TOKEN)

    while True:
        user_input = input("Enter a prompt and # if you want to end: ")
        if user_input.endswith("#"):
            break

        response = chatbot.chat(user_input)
        print("Chatbot response: ", response["response"])

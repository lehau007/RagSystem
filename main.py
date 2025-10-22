# Import ST
from chatbot import Chatbot
from dotenv import load_dotenv
if __name__ == "__main__":
    load_dotenv()
    hf_token = load_dotenv()
    ossapi_key = load_dotenv()
    chatbot = Chatbot(ossapi_key, hf_token)

    while True:
        user_input = input("Enter a prompt and # if you want to end: ")
        if user_input.endswith("#"):
            break

        response = chatbot.chat(user_input)
        print("Chatbot response: ", response["response"])

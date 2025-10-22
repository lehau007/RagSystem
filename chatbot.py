from find_relevant_content import DocumentRetrieval
from openai import OpenAI
import json
from dotenv import load_dotenv

class Chatbot:
    def __init__(self, ossapi_key: str, hf_token: str): 
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=ossapi_key
        )
        self.RagTool = DocumentRetrieval(hf_token)
        self.starting_system_prompt = """You are a knowledgeable assistant, you can decide to use tools base on current and history context and access to a document search tool (rag_tool).

**When to use the rag_tool:**
- User asks about specific policies, regulations, or procedures
- Questions require factual information from documents
- Requests for exact rules, requirements, or guidelines
- Queries about specific criteria, conditions, or standards
- You do not see any information about this in conversation history.

**When NOT to use the rag_tool:**
- General greetings or casual conversation
- Questions you can confidently answer from general knowledge
- Simple clarifications or follow-up questions about previous answers
- User asks about knowledge which appeared in conversation history.

**Important rules:**
1. If you use the rag_tool but find no relevant information, clearly state: "I could not find any information about this in the available documents."
2. Always cite sources when using information from documents (include page numbers)
3. Be honest about limitations - don't make up information
4. Use Vietnamese language when the user writes in Vietnamese"""

        self.rag_system_prompt = """Now, you get the information from rag_tool. Go ahead and answer the user.
**Important rules:**
1. If no relevant information, clearly state: "I could not find any information about this in the available documents."
2. Always cite sources when using information from documents (include page numbers)
3. Be honest about limitations - don't make up information
4. Use Vietnamese language when the user writes in Vietnamese"""

        self.conversation_history = []

        # Update the tool description
        self.rag_tool_description = {
            "type": "function",
            "function": {
                "name": "rag_tool",
                "description": "Find the relevant context for user query using keyword matching and semantic similarity search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "The user's question or search query"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of keywords to search for exact matches"
                        },
                        "use_similarity": {
                            "type": "boolean",
                            "description": "Whether to use semantic similarity search (default: True)"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 4)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
    def reset(self):
        del self.conversation_history 
        self.conversation_history = []

    def summize_chat_history(self):
        pass

    def chat(self, user_prompt: str, keywords=None):
        
        messages = []
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "system", "content": self.starting_system_prompt})
        
        try: 
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=[self.rag_tool_description],
                tool_choice="auto",
                temperature=0.7,
                max_tokens=4096
            )
            
            response_message = completion.choices[0].message
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute RAG tool
                rag_result = self.RagTool.process_rag_tool(
                    query=function_args.get('query', user_prompt),
                    keywords=function_args.get('keywords', keywords or []),
                    use_similarity= True, # function_args.get('use_similarity', True),
                    k=function_args.get('k', 4)
                )
                
                # Send RAG results back to GPT
                context_to_send = rag_result['context'] if rag_result['num_results'] > 0 else "No relevant documents found."

                messages[-1]["content"] = self.rag_system_prompt # Update system prompt when get content
                messages.append(response_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "rag_tool",
                    "content": context_to_send
                })
                
                final_completion = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096
                )

                assistant_response = final_completion.choices[0].message.content
                used_rag = True
                num_sources = rag_result['num_results']
            else:
                assistant_response = response_message.content
                used_rag = False
                num_sources = 0
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return {
                "response": assistant_response,
                "used_rag": used_rag,
                "num_sources": num_sources,
                "history_length": len(self.conversation_history)
            }

        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    load_dotenv()
    hf_token = load_dotenv("HF_TOKEN")
    ossapi_key = load_dotenv("OSSAPI_KEY")

    print(hf_token, ossapi_key)
    chatbot = Chatbot(ossapi_key, hf_token)
    print("Test creating chatbot")

    print(chatbot.chat("hello"))

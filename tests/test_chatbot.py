import unittest
from unittest.mock import MagicMock, patch
from core.chatbot import Chatbot

class TestChatbot(unittest.TestCase):

    @patch('core.chatbot.DocumentRetrieval')
    @patch('core.chatbot.OpenAI')
    def test_chatbot_initialization(self, mock_openai, mock_doc_retrieval):
        """Test that the chatbot initializes correctly."""
        mock_openai.return_value = MagicMock()
        mock_doc_retrieval.return_value = MagicMock()
        
        chatbot = Chatbot(ossapi_key="fake_key", hf_token="fake_token")
        self.assertIsNotNone(chatbot)
        mock_openai.assert_called_with(base_url="https://integrate.api.nvidia.com/v1", api_key="fake_key")
        mock_doc_retrieval.assert_called_with("fake_token")

    @patch('core.chatbot.DocumentRetrieval')
    @patch('core.chatbot.OpenAI')
    def test_chat_no_rag(self, mock_openai, mock_doc_retrieval):
        """Test a simple chat interaction that doesn't use the RAG tool."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_doc_retrieval.return_value = MagicMock()

        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Hello from the mock AI!"
        mock_client.chat.completions.create.return_value = mock_response

        # Initialize chatbot and chat
        chatbot = Chatbot(ossapi_key="fake_key", hf_token="fake_token")
        result = chatbot.chat("Hello")

        # Assertions
        self.assertEqual(result['response'], "Hello from the mock AI!")
        self.assertFalse(result['used_rag'])
        self.assertEqual(result['num_sources'], 0)
        mock_client.chat.completions.create.assert_called_once()

    @patch('core.chatbot.DocumentRetrieval')
    @patch('core.chatbot.OpenAI')
    def test_chat_with_rag(self, mock_openai, mock_doc_retrieval):
        """Test a chat interaction that triggers the RAG tool."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_rag_tool = MagicMock()
        mock_rag_tool.process_rag_tool.return_value = {
            'context': 'This is relevant context.',
            'num_results': 1,
            'results': []
        }
        mock_doc_retrieval.return_value = mock_rag_tool

        # Mock the initial OpenAI response to call the tool
        mock_tool_call_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = '{"query": "some query"}'
        mock_tool_call.id = "tool_call_123"
        mock_tool_call_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock the final OpenAI response after getting tool context
        mock_final_response = MagicMock()
        mock_final_response.choices[0].message.content = "This is the final answer based on context."

        # Set the side_effect for multiple calls
        mock_client.chat.completions.create.side_effect = [
            mock_tool_call_response,
            mock_final_response
        ]

        # Initialize chatbot and chat
        chatbot = Chatbot(ossapi_key="fake_key", hf_token="fake_token")
        result = chatbot.chat("Tell me something that requires context")

        # Assertions
        self.assertEqual(result['response'], "This is the final answer based on context.")
        self.assertTrue(result['used_rag'])
        self.assertEqual(result['num_sources'], 1)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        mock_rag_tool.process_rag_tool.assert_called_once()

if __name__ == '__main__':
    unittest.main()

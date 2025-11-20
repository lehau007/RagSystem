import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# It's important to patch the chatbot before it's imported by the api.main
# We create a mock Chatbot that can be controlled in the tests
mock_chatbot_instance = MagicMock()

def get_mock_chatbot(*args, **kwargs):
    return mock_chatbot_instance

# Patch the Chatbot class in the core.chatbot module
# This will affect the instance created in api.main
patcher = patch('core.chatbot.Chatbot', new=get_mock_chatbot)
patcher.start()

# Now we can import the app
from api.main import app

# Stop the patcher after the app is imported and set up
# to avoid it affecting other tests if this file is imported elsewhere.
patcher.stop()


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance for the API."""
    with TestClient(app) as c:
        yield c

def test_chat_endpoint_success(client):
    """Test the /chat endpoint with a successful response."""
    # Configure the mock to return a specific value
    mock_response = {
        "response": "This is a mock response.",
        "used_rag": False,
        "num_sources": 0,
        "history_length": 2
    }
    mock_chatbot_instance.chat.return_value = mock_response
    
    # Send a request to the API
    response = client.post("/chat", params={"user_input": "Hello"})
    
    # Assertions
    assert response.status_code == 200
    assert response.json() == mock_response
    # Verify that the chat method was called with the correct argument
    mock_chatbot_instance.chat.assert_called_with("Hello")

def test_chat_endpoint_empty_input(client):
    """Test the /chat endpoint with empty user input."""
    mock_response = {
        "response": "Empty input received.",
        "used_rag": False,
        "num_sources": 0,
        "history_length": 2
    }
    mock_chatbot_instance.chat.return_value = mock_response

    response = client.post("/chat", params={"user_input": ""})
    
    assert response.status_code == 200
    assert response.json() == mock_response
    mock_chatbot_instance.chat.assert_called_with("")

def test_chat_endpoint_internal_error(client):
    """Test the /chat endpoint when the chatbot raises an exception."""
    # Configure the mock to raise an exception
    error_message = "Internal chatbot error"
    mock_chatbot_instance.chat.side_effect = Exception(error_message)
    
    # It's generally better for the endpoint to handle the exception
    # and return a proper HTTP error. For this example, we assume
    # FastAPI's default exception handling will catch it.
    # A more robust implementation would have a try-except block in the endpoint.
    
    # In a real app, you'd expect a 500 error.
    # The default TestClient behavior might raise the exception directly.
    with pytest.raises(Exception) as excinfo:
         client.post("/chat", params={"user_input": "trigger error"})
    
    assert error_message in str(excinfo.value)
    mock_chatbot_instance.chat.assert_called_with("trigger error")


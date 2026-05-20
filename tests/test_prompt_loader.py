import sys
from unittest.mock import MagicMock, patch


class TestLoadLocalYaml:
    def test_decompose_query_template_has_query_placeholder(self):
        from core.prompt_loader import load_prompt
        template = load_prompt("decompose_query")
        assert "{query}" in template

    def test_synthesize_response_template_has_both_placeholders(self):
        from core.prompt_loader import load_prompt
        template = load_prompt("synthesize_response")
        assert "{query}" in template
        assert "{contexts}" in template

    def test_no_hub_path_always_uses_local(self):
        """With no hub_path, local YAML is used regardless of LANGSMITH_ENABLED."""
        with patch("config.settings.LANGSMITH_ENABLED", True):
            from core.prompt_loader import load_prompt
            template = load_prompt("decompose_query")
        assert "{query}" in template


class TestHubFallback:
    def test_hub_called_when_langsmith_enabled(self):
        """When LangSmith is enabled + hub_path given, hub.pull is attempted."""
        mock_hub = MagicMock()
        mock_hub.pull.return_value.template = "Hub template: {query}"
        mock_langchain = MagicMock()
        mock_langchain.hub = mock_hub

        with patch.dict(sys.modules, {"langchain": mock_langchain}), \
             patch("config.settings.LANGSMITH_ENABLED", True):
            from core.prompt_loader import load_prompt
            template = load_prompt("decompose_query", hub_path="user/decompose")

        assert template == "Hub template: {query}"
        mock_hub.pull.assert_called_once_with("user/decompose")

    def test_falls_back_to_local_when_hub_raises(self):
        """If hub.pull raises any exception, falls back to local YAML."""
        mock_hub = MagicMock()
        mock_hub.pull.side_effect = Exception("Hub unreachable")
        mock_langchain = MagicMock()
        mock_langchain.hub = mock_hub

        with patch.dict(sys.modules, {"langchain": mock_langchain}), \
             patch("config.settings.LANGSMITH_ENABLED", True):
            from core.prompt_loader import load_prompt
            template = load_prompt("decompose_query", hub_path="user/decompose")

        assert "{query}" in template

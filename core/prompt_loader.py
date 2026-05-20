import yaml
from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).parent.parent / "langsmith_prompts"


def load_prompt(prompt_name: str, hub_path: Optional[str] = None) -> str:
    """Load prompt template — tries LangSmith Hub first, falls back to local YAML."""
    from config.settings import LANGSMITH_ENABLED
    if LANGSMITH_ENABLED and hub_path:
        try:
            from langchain import hub
            return hub.pull(hub_path).template
        except Exception:
            pass
    with open(PROMPTS_DIR / f"{prompt_name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["template"]

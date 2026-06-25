"""LLM factory.

A thin wrapper around LangChain's ``init_chat_model`` (the same approach as the
Opsfleet client suggested in the assignment, but provider-agnostic). The active
provider and model come from the environment, so switching from OpenAI to Gemini
is a config change, not a code change.
"""

from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.settings import settings

# OpenRouter is OpenAI-compatible, so we reuse the openai integration with its
# base url instead of a dedicated provider.
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Build the chat model for the provider set in the environment.

    temperature=0 keeps SQL generation and reports stable and repeatable.
    """
    kwargs: dict = {
        "model": settings.llm_model,
        "temperature": 0,
        "api_key": settings.api_key(),
    }

    if settings.llm_provider == "openrouter":
        kwargs["model_provider"] = "openai"
        kwargs["base_url"] = _OPENROUTER_BASE
    else:
        kwargs["model_provider"] = settings.llm_provider

    return init_chat_model(**kwargs)

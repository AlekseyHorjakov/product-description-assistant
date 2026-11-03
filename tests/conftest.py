"""Общие фикстуры для тестов."""

import pytest

from config import Settings


@pytest.fixture
def settings() -> Settings:
    """Настройки, которые никуда не звонят: сеть в тестах не используется."""
    return Settings(
        api_key="test-key",
        base_url="http://127.0.0.1:11434/v1",
        model="test-model",
        temperature=0.2,
        max_tokens=1000,
        timeout=5.0,
        json_mode=True,
    )

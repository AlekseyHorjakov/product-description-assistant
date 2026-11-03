"""Проверки клиента LLM: обработка ключа, таймаута, ответов и ошибок API.

Клиент SDK подменяется простым объектом-заглушкой, поэтому сеть не нужна.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from config import Settings
from errors import LLMError
from llm_client import create_client, request_completion


def fake_client(create):
    """Объект с той же «формой», что и настоящий клиент OpenAI."""
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def fake_response(content: str, choices: bool = True):
    if not choices:
        return SimpleNamespace(choices=[], usage=None)
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


def http_request() -> httpx.Request:
    return httpx.Request("POST", "http://test/v1/chat/completions")


def test_missing_api_key(settings):
    without_key = Settings(
        api_key="",
        base_url=settings.base_url,
        model=settings.model,
        temperature=0.2,
        max_tokens=100,
        timeout=5.0,
        json_mode=True,
    )

    with pytest.raises(LLMError) as error:
        create_client(without_key)

    assert "API-ключ" in str(error.value)


def test_authentication_error_is_explained(settings):
    def create(**_kwargs):
        raise AuthenticationError(
            "invalid api key",
            response=httpx.Response(401, request=http_request()),
            body=None,
        )

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "401" in str(error.value)


def test_timeout_is_explained(settings):
    def create(**_kwargs):
        raise APITimeoutError(request=http_request())

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "не ответила" in str(error.value)


def test_connection_error_is_explained(settings):
    def create(**_kwargs):
        raise APIConnectionError(request=http_request())

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "подключиться" in str(error.value)


def test_rate_limit_is_explained(settings):
    def create(**_kwargs):
        raise RateLimitError(
            "too many requests",
            response=httpx.Response(429, request=http_request()),
            body=None,
        )

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "429" in str(error.value)


def test_server_error_status_is_explained(settings):
    def create(**_kwargs):
        raise APIStatusError(
            "internal error",
            response=httpx.Response(500, request=http_request()),
            body=None,
        )

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "HTTP 500" in str(error.value)


def test_unexpected_exception_is_wrapped(settings):
    def create(**_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "RuntimeError" in str(error.value)


def test_empty_content_is_reported(settings):
    def create(**_kwargs):
        return fake_response("   ")

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "пустой ответ" in str(error.value)


def test_empty_choices_is_reported(settings):
    def create(**_kwargs):
        return fake_response("", choices=False)

    with pytest.raises(LLMError) as error:
        request_completion([], settings, client=fake_client(create))

    assert "choices" in str(error.value)


def test_successful_answer_is_returned(settings):
    def create(**_kwargs):
        return fake_response("{\"ok\": true}")

    assert request_completion([], settings, client=fake_client(create)) == '{"ok": true}'


def test_json_mode_is_passed_to_api(settings):
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return fake_response("{}")

    request_completion([{"role": "user", "content": "привет"}], settings, client=fake_client(create))

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.2


def test_json_mode_can_be_disabled(settings):
    captured: dict = {}
    without_json = Settings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        temperature=0.2,
        max_tokens=100,
        timeout=5.0,
        json_mode=False,
    )

    def create(**kwargs):
        captured.update(kwargs)
        return fake_response("{}")

    request_completion([], without_json, client=fake_client(create))

    assert "response_format" not in captured

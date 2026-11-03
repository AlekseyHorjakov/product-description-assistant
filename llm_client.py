"""Клиент LLM: единственное место, где приложение ходит в сеть.

Используется официальный SDK openai: DeepSeek, Ollama и большинство других
провайдеров предоставляют OpenAI-совместимый API, поэтому смена провайдера —
это смена трёх переменных в .env, а не переписывание кода.
"""

import logging
from urllib.parse import urlparse

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from config import Settings
from errors import LLMError

logger = logging.getLogger(__name__)


def _is_local(base_url: str) -> bool:
    """Локальный ли адрес (Ollama, llama.cpp и т. п.)."""
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def create_client(settings: Settings) -> OpenAI:
    """Создаёт клиент SDK. Ключ берётся только из настроек, не из кода."""
    if not settings.api_key:
        raise LLMError(
            "Не задан API-ключ. Скопируйте .env.example в .env и укажите LLM_API_KEY."
        )

    # Для локальной модели отключаем системный прокси: иначе прокси Windows
    # перехватывает запрос к 127.0.0.1 и возвращает ошибку 502.
    http_client = httpx.Client(trust_env=not _is_local(settings.base_url))

    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        max_retries=2,
        http_client=http_client,
    )


def request_completion(
    messages: list[dict[str, str]],
    settings: Settings,
    client: OpenAI | None = None,
) -> str:
    """Отправляет сообщения в LLM и возвращает текст ответа.

    Все сетевые и API-ошибки превращаются в LLMError с понятным текстом,
    чтобы интерфейс не показывал пользователю трейсбек.
    """
    client = client or create_client(settings)

    params: dict = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if settings.json_mode:
        # Просим модель вернуть именно JSON-объект, а не текст с разметкой.
        params["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**params)
    except AuthenticationError as exc:
        raise LLMError(
            "LLM отклонила API-ключ (ошибка 401). Проверьте LLM_API_KEY в файле .env."
        ) from exc
    except APITimeoutError as exc:
        raise LLMError(
            f"LLM не ответила за {settings.timeout:.0f} секунд. "
            "Повторите запрос или увеличьте LLM_TIMEOUT в .env."
        ) from exc
    except APIConnectionError as exc:
        raise LLMError(
            f"Не удалось подключиться к LLM по адресу {settings.base_url}. "
            "Проверьте интернет, LLM_BASE_URL и доступность сервера."
        ) from exc
    except RateLimitError as exc:
        raise LLMError("LLM ограничила частоту запросов (ошибка 429). Подождите и повторите.") from exc
    except APIStatusError as exc:
        raise LLMError(f"LLM вернула ошибку HTTP {exc.status_code}: {_short(exc.message)}") from exc
    except APIError as exc:
        raise LLMError(f"Ошибка LLM API: {_short(exc.message)}") from exc
    except Exception as exc:  # сеть, SSL, несовместимый провайдер
        logger.exception("Неожиданная ошибка при запросе к LLM")
        raise LLMError(f"Не удалось получить ответ от LLM: {type(exc).__name__}.") from exc

    if not response.choices:
        raise LLMError("LLM вернула ответ без вариантов (choices пуст).")

    content = response.choices[0].message.content or ""
    if not content.strip():
        raise LLMError(
            "LLM вернула пустой ответ. Попробуйте ещё раз или выберите другую модель."
        )

    if response.usage:
        logger.info(
            "LLM ok: model=%s, prompt_tokens=%s, completion_tokens=%s",
            settings.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
    return content


def _short(text: str | None, limit: int = 300) -> str:
    text = (text or "").strip()
    return text[:limit] + ("…" if len(text) > limit else "")

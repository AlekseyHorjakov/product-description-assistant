"""Настройки приложения: читаются из переменных окружения (файла .env).

Здесь нет ни одного «зашитого» ключа: всё, что зависит от окружения,
настраивается через .env. Образец лежит в .env.example.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from errors import AppError

# значения по умолчанию: DeepSeek API (совместим с OpenAI SDK)
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


@dataclass(frozen=True)
class Settings:
    """Одна неизменяемая «коробка» с настройками."""

    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    json_mode: bool


def _read_text(name: str, default: str) -> str:
    """Читает строковую переменную окружения и убирает пробелы по краям."""
    value = os.getenv(name, "").strip()
    return value or default


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise AppError(f"Переменная {name} должна быть целым числом, а сейчас «{raw}».") from exc


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise AppError(f"Переменная {name} должна быть числом, а сейчас «{raw}».") from exc


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def load_settings() -> Settings:
    """Загружает .env и собирает настройки.

    load_dotenv() не перезаписывает уже заданные переменные окружения,
    поэтому значения из .env можно временно переопределить из терминала.
    """
    load_dotenv()

    return Settings(
        api_key=_read_text("LLM_API_KEY", ""),
        base_url=_read_text("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        model=_read_text("LLM_MODEL", DEFAULT_MODEL),
        temperature=_read_float("LLM_TEMPERATURE", 0.2),
        max_tokens=_read_int("LLM_MAX_TOKENS", 2000),
        timeout=_read_float("LLM_TIMEOUT", 60.0),
        json_mode=_read_bool("LLM_JSON_MODE", True),
    )

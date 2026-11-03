"""Модели данных и разбор пользовательского ввода.

Здесь два «мира»:

1. ProductInput — то, что пришло от менеджера (форма или JSON).
   Pydantic проверяет типы и обязательные поля и не пропускает мусор.

2. ProductCard — то, что вернула LLM.
   Это же структура используется для проверок результата и для выгрузки.

Ограничения из ТЗ (длины, количество запросов) вынесены в константы,
чтобы менять их в одном месте.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from errors import ProductDataError

# --- Ограничения из технического задания ---------------------------------
SEO_TITLE_MAX = 70
SEO_DESCRIPTION_MIN = 800
SEO_DESCRIPTION_MAX = 1200
SEARCH_QUERIES_MIN = 3
SEARCH_QUERIES_MAX = 5

# Единая пометка для отсутствующих сведений (требование ТЗ)
NO_DATA = "Данных нет"

# Человеческие названия полей для сообщений об ошибках
FIELD_LABELS = {
    "name": "название товара",
    "manufacturer": "производитель",
    "specs": "технические характеристики",
    "description": "описание",
}


class ProductInput(BaseModel):
    """Данные товара, введённые менеджером."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    name: str = Field(min_length=2, max_length=200)
    manufacturer: str = Field(min_length=1, max_length=200)
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = Field(default="", max_length=20000)

    @field_validator("name", "manufacturer", "description", mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> Any:
        """Число в поле текста превращаем в строку, остальное отдаём Pydantic."""
        if value is None:
            return ""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("specs", mode="before")
    @classmethod
    def _clean_specs(cls, value: Any) -> dict[str, str]:
        """Приводим характеристики к словарю {название: значение} из строк."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('ожидается объект вида {"характеристика": "значение"}')

        cleaned: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("название характеристики должно быть непустой строкой")
            if item is None:
                continue  # пустое значение просто не включаем
            if isinstance(item, (dict, list, bool)):
                raise ValueError(f"значение характеристики «{key}» должно быть строкой или числом")
            text = str(item).strip()
            if text:
                cleaned[key.strip()] = text
        return cleaned


class KeySpec(BaseModel):
    """Одна характеристика в готовой карточке."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=500)


class ProductCard(BaseModel):
    """Готовая карточка товара — ровно те разделы, что требует ТЗ."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    seo_title: str = Field(min_length=1, max_length=300)
    seo_description: str = Field(min_length=1, max_length=8000)
    key_specs: list[KeySpec] = Field(min_length=1, max_length=40)
    search_queries: list[str] = Field(default_factory=list)
    commercial_block: str = Field(min_length=1, max_length=4000)
    missing_data: list[str] = Field(default_factory=list)

    @field_validator("key_specs", mode="before")
    @classmethod
    def _normalize_specs(cls, value: Any) -> Any:
        """Принимаем и объект, и список строк вида «Название: значение»."""
        if isinstance(value, dict):
            return [{"name": key, "value": item} for key, item in value.items()]
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str):
                    name, separator, raw_value = item.partition(":")
                    result.append({"name": name, "value": raw_value if separator else NO_DATA})
                else:
                    result.append(item)
            return result
        return value

    @field_validator("search_queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        """Строку превращаем в список, числа — в строки, None — в пустой список."""
        if value is None:
            return []
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    @field_validator("missing_data", mode="before")
    @classmethod
    def _normalize_missing(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value


# --- Разбор пользовательского ввода ---------------------------------------

SPEC_SEPARATOR = re.compile(r"[:=\t]")
BOM = "\ufeff"


def parse_json_text(text: str) -> dict[str, Any]:
    """Разбирает JSON из текстового поля и объясняет ошибку по-русски."""
    if not text or not text.strip():
        raise ProductDataError("JSON не введён: вставьте данные товара.")

    cleaned = text.lstrip(BOM)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProductDataError(
            f"Некорректный JSON: строка {exc.lineno}, колонка {exc.colno} — {exc.msg}. "
            "Проверьте кавычки, запятые и закрывающие скобки."
        ) from exc

    if not isinstance(data, dict):
        raise ProductDataError(
            'Ожидается JSON-объект с полями "name", "manufacturer", "specs", "description".'
        )
    return data


def parse_product(raw: dict[str, Any] | str) -> ProductInput:
    """Единая точка входа для формы и JSON: и то и другое приходит сюда."""
    data = parse_json_text(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        raise ProductDataError('Ожидается объект с данными товара.')

    try:
        return ProductInput.model_validate(data)
    except ValidationError as exc:
        raise _validation_error(exc) from exc


def _validation_error(exc: ValidationError) -> ProductDataError:
    """Переводим ошибки Pydantic в короткие понятные фразы."""
    lines: list[str] = []
    for error in exc.errors():
        field = str(error["loc"][0]) if error["loc"] else "данные"
        label = FIELD_LABELS.get(field, field)
        error_type = error["type"]
        message = str(error.get("msg", "")).removeprefix("Value error, ")

        if error_type == "missing":
            lines.append(f"поле «{label}» обязательно для заполнения")
        elif error_type == "string_too_short":
            lines.append(f"поле «{label}» не может быть пустым")
        elif error_type == "string_too_long":
            limit = error.get("ctx", {}).get("max_length", "")
            lines.append(f"поле «{label}» слишком длинное (максимум {limit} символов)")
        else:
            lines.append(f"поле «{label}»: {message}")

    message = "Проверьте данные товара:\n" + "\n".join(f"— {line}" for line in lines)
    return ProductDataError(message)


def specs_from_text(text: str) -> dict[str, str]:
    """Превращает многострочный текст «название: значение» в словарь.

    Поддерживаются разделители «:», «=» и табуляция. Строки без разделителя
    считаются ошибкой: лучше показать её сразу, чем молча потерять данные.
    """
    specs: dict[str, str] = {}
    problems: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip().lstrip("-•*").strip()
        if not line:
            continue

        parts = SPEC_SEPARATOR.split(line, maxsplit=1)
        if len(parts) == 1:
            problems.append(f"строка {line_number}: «{line[:40]}» — нет разделителя «:»")
            continue

        name, value = parts[0].strip(), parts[1].strip()
        if not name:
            problems.append(f"строка {line_number}: пустое название характеристики")
        elif not value:
            problems.append(f"строка {line_number}: у характеристики «{name}» пустое значение")
        else:
            specs[name] = value

    if problems:
        raise ProductDataError(
            "Не удалось разобрать характеристики:\n" + "\n".join(f"— {item}" for item in problems)
        )
    return specs

"""Сквозной сценарий: данные -> промпт -> LLM -> разбор -> проверки -> результат.

Модель — не единственный «контролёр качества». После ответа приложение
самостоятельно проверяет:
* длины SEO-заголовка и описания и количество запросов (ограничения ТЗ);
* числа в ответе — все они должны встречаться во входных данных;
* единицы измерения — не подменены ли они моделью;
* рекламные и неподтверждённые утверждения (сертификаты, гарантия и т. п.).

Если проверки что-то нашли, делается один повторный запрос с описанием
нарушений. Если нарушения остались — результат всё равно показывается,
но с предупреждениями, чтобы менеджер не публиковал черновик вслепую.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from config import Settings
from errors import ResponseFormatError
from llm_client import request_completion
from models import (
    NO_DATA,
    SEARCH_QUERIES_MAX,
    SEARCH_QUERIES_MIN,
    SEO_DESCRIPTION_MAX,
    SEO_DESCRIPTION_MIN,
    SEO_TITLE_MAX,
    ProductCard,
    ProductInput,
)
from prompts import SYSTEM_PROMPT, build_retry_prompt, build_user_prompt

# Одна основная попытка и один повторный запрос, если проверки нашли проблемы.
MAX_ATTEMPTS = 2

# Функция, которая отправляет messages в модель и возвращает текст.
# Аннотация нужна, чтобы в тестах можно было подставить заглушку.
CompletionFn = Callable[[list[dict[str, str]], Settings], str]

# Число и то, что написано сразу после него: «1,6 МПа», «100 мм», «24В».
# Третья группа — кандидат в единицы измерения, её проверяет _parse_unit().
NUMBER_RE = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)(\s*)([%°]?[A-Za-zА-Яа-яЁё]{0,10})"
)

# Известные единицы измерения. Слово вне этого списка единицей не считается:
# иначе «DM 02 предназначен» давало бы «единицу» вроде «предназначен».
KNOWN_UNITS = {
    # давление
    "па", "кпа", "мпа", "бар", "psi", "атм",
    # длина
    "мм", "см", "м", "км", "мкм", "дюйм",
    # масса
    "г", "кг", "т", "мг",
    # электричество
    "в", "кв", "мв", "а", "ма", "вт", "квт",
    # частота
    "гц", "кгц", "мгц",
    # время
    "с", "мин", "ч", "час", "часа",
    # объём
    "л", "мл", "м3",
    # прочее
    "шт", "об", "ip", "%", "°", "°c", "°f",
    # латиница
    "v", "mv", "kv", "a", "ma", "w", "kw", "hz", "khz", "mhz",
    "bar", "mpa", "kpa", "pa", "kg", "mm", "cm", "m", "km", "l", "ml", "s", "min", "h",
}

# Фразы, которых не должно быть, если их нет во входных данных.
FORBIDDEN_CLAIMS = {
    r"сертифик\w*": "сертификат",
    r"гаранти\w*": "гарантия",
    r"поверк\w*": "поверка",
    r"\bгост\b": "ГОСТ",
    r"патент\w*": "патент",
    r"награ\w*": "награда",
    r"лучш\w*": "лучший",
    r"\bсамы\w*": "самый",
    r"лидер\w*": "лидер",
    r"№\s*1\b": "№1",
    r"бесплатн\w*": "бесплатно",
    r"скидк\w*": "скидка",
    r"стандарт\w*": "стандарт",
    r"надежн\w*": "надёжность",
}

# Служебные слова не участвуют в сравнении названий характеристик.
STOPWORDS = {"для", "или", "при", "без", "тип", "вид", "этот", "какой", "который"}

# «не» + слово: ловим смысловые перевороты («неагрессивные» -> «агрессивные»).
NEGATED_WORD_RE = re.compile(r"\bне([а-яёa-z]{6,})", re.IGNORECASE)

CARD_FIELDS = {
    "seo_title": "SEO-заголовок",
    "seo_description": "SEO-описание",
    "key_specs": "ключевые характеристики",
    "search_queries": "поисковые запросы",
    "commercial_block": "коммерческий блок",
    "missing_data": "отсутствующие данные",
}


@dataclass
class GenerationResult:
    """Результат генерации вместе с замечаниями проверок."""

    card: ProductCard
    warnings: list[str]
    attempts: int
    model: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        data = self.card.model_dump()
        data["warnings"] = self.warnings
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        card = self.card
        lines = [
            "## SEO-заголовок",
            card.seo_title,
            "",
            "## SEO-описание",
            card.seo_description,
            "",
            "## Ключевые характеристики",
            "| Характеристика | Значение |",
            "| --- | --- |",
        ]
        lines += [f"| {_md_cell(spec.name)} | {_md_cell(spec.value)} |" for spec in card.key_specs]
        lines += ["", "## Поисковые запросы"]
        lines += [f"- {query}" for query in card.search_queries]
        lines += ["", "## Коммерческий блок для менеджера", card.commercial_block]

        if card.missing_data:
            lines += ["", "## Требуется уточнить"]
            lines += [f"- {item}" for item in card.missing_data]

        if self.warnings:
            lines += ["", "## Замечания автоматических проверок"]
            lines += [f"- {item}" for item in self.warnings]

        return "\n".join(lines)


def generate_card(
    product: ProductInput,
    settings: Settings,
    completion_fn: CompletionFn = request_completion,
) -> GenerationResult:
    """Генерирует карточку и проверяет ответ модели.

    completion_fn подменяется в тестах на функцию-заглушку, поэтому
    тесты не ходят в сеть.
    """
    prompt = build_user_prompt(product)
    card: ProductCard | None = None
    problems: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw_answer = completion_fn(messages, settings)

        try:
            card = parse_card_response(raw_answer)
        except ResponseFormatError as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            problems = [str(exc)]
            prompt = build_retry_prompt(product, problems)
            continue

        problems = find_problems(card, product)
        if not problems:
            break
        if attempt < MAX_ATTEMPTS:
            prompt = build_retry_prompt(product, problems)

    if card is None:  # защита от неверного изменения логики
        raise ResponseFormatError("Модель не вернула карточку товара.")

    warnings = find_soft_warnings(card, product) + problems
    return GenerationResult(
        card=card,
        warnings=warnings,
        attempts=attempt,
        model=settings.model,
        prompt=prompt,
    )


# --- Разбор ответа модели --------------------------------------------------

def parse_card_response(raw_answer: str) -> ProductCard:
    """Достаёт JSON из ответа и проверяет структуру через Pydantic."""
    data = extract_json(raw_answer)
    try:
        return ProductCard.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(_describe_error(error) for error in exc.errors())
        raise ResponseFormatError(
            f"Ответ модели не соответствует структуре карточки: {details}."
        ) from exc


def extract_json(raw_answer: str) -> dict[str, Any]:
    """Ищет JSON-объект в ответе, даже если модель добавила обёртку."""
    text = (raw_answer or "").strip()
    if not text:
        raise ResponseFormatError("Модель вернула пустой ответ.")

    text = _strip_code_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ResponseFormatError(
                "Модель вернула не JSON. Повторите запрос или выберите другую модель."
            )
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ResponseFormatError(
                f"Модель вернула JSON с синтаксической ошибкой: {exc.msg} "
                f"(строка {exc.lineno})."
            ) from exc

    if not isinstance(data, dict):
        raise ResponseFormatError("Модель вернула JSON, но это не объект с полями карточки.")
    return data


def _strip_code_fence(text: str) -> str:
    """Убирает ```json ... ```, если модель всё-таки добавила разметку."""
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```[A-Za-z]*\s*", "", text)
    return re.sub(r"\s*```\s*$", "", text).strip()


def _describe_error(error: dict[str, Any]) -> str:
    field = str(error["loc"][0]) if error["loc"] else "ответ"
    label = CARD_FIELDS.get(field, field)
    message = str(error.get("msg", "")).removeprefix("Value error, ")
    if error.get("type") == "missing":
        return f"нет поля «{label}»"
    if error.get("type") == "too_short" and field == "key_specs":
        return "список «ключевые характеристики» пуст"
    return f"поле «{label}»: {message}"


# --- Жёсткие проверки (вызывают повторный запрос) --------------------------

def find_problems(card: ProductCard, product: ProductInput) -> list[str]:
    """Нарушения, которые можно исправить повторным запросом к модели."""
    problems: list[str] = []

    title_length = len(card.seo_title)
    if title_length > SEO_TITLE_MAX:
        problems.append(
            f"SEO-заголовок длиннее {SEO_TITLE_MAX} символов: сейчас {title_length}. Сократи его."
        )

    description_length = len(card.seo_description)
    if description_length < SEO_DESCRIPTION_MIN:
        problems.append(
            f"SEO-описание короче {SEO_DESCRIPTION_MIN} символов: сейчас {description_length}. "
            "Доведи объём до 800–1200 символов, используя только данные из запроса."
        )
    elif description_length > SEO_DESCRIPTION_MAX:
        problems.append(
            f"SEO-описание длиннее {SEO_DESCRIPTION_MAX} символов: сейчас {description_length}. "
            "Сократи текст."
        )

    queries_count = len(card.search_queries)
    if not SEARCH_QUERIES_MIN <= queries_count <= SEARCH_QUERIES_MAX:
        problems.append(
            f"Поисковых запросов должно быть от {SEARCH_QUERIES_MIN} до {SEARCH_QUERIES_MAX}, "
            f"сейчас {queries_count}."
        )

    if not card.key_specs:
        problems.append("Список ключевых характеристик пуст.")

    problems += _unknown_numbers(card, product)
    problems += _forbidden_claims(card, product)
    problems += _negation_flips(card, product)
    return _dedupe(problems)


def _unknown_numbers(card: ProductCard, product: ProductInput) -> list[str]:
    """Ищет числа, которых нет во входных данных (признак выдумки)."""
    known = _collect_numbers(_product_text(product))
    problems: list[str] = []

    for label, text in _output_fields(card):
        for raw_number, spaces, raw_unit in NUMBER_RE.findall(text):
            number = _normalize_number(raw_number)
            if number not in known:
                shown = f"{raw_number} {raw_unit}".strip() if raw_unit else raw_number
                problems.append(
                    f"В поле «{label}» есть значение «{shown}», которого нет во входных данных. "
                    "Убери его или замени на «Данных нет»."
                )
                continue

            unit = _parse_unit(raw_unit, spaces)
            if unit and unit not in known[number] and "" not in known[number]:
                expected = ", ".join(sorted(item for item in known[number] if item))
                problems.append(
                    f"В поле «{label}» у значения «{raw_number}» единица «{unit}» не совпадает "
                    f"с исходными данными ({expected or 'без единицы'}). Исправь единицу измерения."
                )
    return problems


def _forbidden_claims(card: ProductCard, product: ProductInput) -> list[str]:
    """Ловит сертификаты, гарантии и рекламные штампы, которых нет во входных данных."""
    source = _normalize_value(_product_text(product))
    problems: list[str] = []

    for pattern, title in FORBIDDEN_CLAIMS.items():
        if re.search(pattern, source):
            continue  # во входных данных это упоминается — утверждение допустимо
        for label, text in _output_fields(card):
            if re.search(pattern, _normalize_value(text)):
                problems.append(
                    f"В поле «{label}» упоминается «{title}», хотя во входных данных этого нет. "
                    "Убери утверждение."
                )
                break
    return problems


def _negation_flips(card: ProductCard, product: ProductInput) -> list[str]:
    """Ловит потерю отрицания: «неагрессивные» -> «агрессивные».

    Смысл характеристики меняется на противоположный, хотя числа и единицы
    остаются на месте, поэтому отдельная проверка обязательна.
    """
    source = _normalize_value(_product_text(product))
    problems: list[str] = []

    for match in NEGATED_WORD_RE.finditer(source):
        positive = match.group(1)
        for label, text in _output_fields(card):
            if _has_unnegated(_normalize_value(text), positive):
                problems.append(
                    f"В поле «{label}» слово «{positive}» стоит без отрицания, хотя во входных "
                    f"данных сказано «не{positive}». Проверь смысл и верни отрицание."
                )
                break
    return problems


def _has_unnegated(text: str, word: str) -> bool:
    """Есть ли в тексте слово без «не» перед ним."""
    for match in re.finditer(re.escape(word), text):
        prefix = text[max(0, match.start() - 3) : match.start()]
        if not prefix.strip().endswith("не"):
            return True
    return False


# --- Мягкие проверки (только предупреждения) -------------------------------
def find_soft_warnings(card: ProductCard, product: ProductInput) -> list[str]:
    """Замечания, из-за которых не стоит повторять запрос."""
    warnings: list[str] = []
    warnings += _input_data_gaps(product)
    warnings += _missing_input_specs(card, product)
    warnings += _unknown_spec_names(card, product)
    return _dedupe(warnings)


def _input_data_gaps(product: ProductInput) -> list[str]:
    """Что менеджер не заполнил ещё до обращения к модели."""
    warnings: list[str] = []
    if not product.specs:
        warnings.append("Во входных данных нет технических характеристик: карточка будет неполной.")

    description = product.description.strip()
    if not description:
        warnings.append(
            "Не указано описание или выдержка из паспорта: модель опирается только на характеристики."
        )
    elif len(description) < 150:
        warnings.append(
            f"Описание короткое ({len(description)} символов): данных может не хватить для полного текста."
        )
    return warnings


def _missing_input_specs(card: ProductCard, product: ProductInput) -> list[str]:
    """Проверяет, что характеристики из входных данных попали в карточку."""
    card_text = "\n".join(text for _, text in _output_fields(card))
    card_numbers = _collect_numbers(card_text)
    card_normalized = _normalize_value(card_text)

    warnings: list[str] = []
    for name, value in product.specs.items():
        numbers = [_normalize_number(raw) for raw, _, _ in NUMBER_RE.findall(value)]
        if numbers:
            present = all(number in card_numbers for number in numbers)
        else:
            present = _normalize_value(value) in card_normalized

        if not present:
            warnings.append(
                f"Характеристика «{name}: {value}» из входных данных не попала в карточку — добавьте вручную."
            )
    return warnings


def _unknown_spec_names(card: ProductCard, product: ProductInput) -> list[str]:
    """Ищет характеристики, название которых не встречается во входных данных."""
    input_tokens = _tokens(_product_text(product))
    warnings: list[str] = []

    for spec in card.key_specs:
        if spec.value.strip().lower() == NO_DATA.lower():
            continue
        name_tokens = _tokens(spec.name)
        if name_tokens and not _tokens_match(name_tokens, input_tokens):
            warnings.append(
                f"Характеристика «{spec.name}» не встречается во входных данных — "
                "возможно, модель добавила её сама. Проверьте."
            )
    return warnings


# --- Вспомогательные функции ----------------------------------------------

def _product_text(product: ProductInput) -> str:
    parts = [product.name, product.manufacturer, product.description]
    parts += [f"{name}: {value}" for name, value in product.specs.items()]
    return "\n".join(parts)


def _output_fields(card: ProductCard) -> list[tuple[str, str]]:
    fields = [
        ("SEO-заголовок", card.seo_title),
        ("SEO-описание", card.seo_description),
        ("Коммерческий блок", card.commercial_block),
    ]
    fields += [(f"Характеристика «{spec.name}»", spec.value) for spec in card.key_specs]
    fields += [(f"Поисковый запрос «{query}»", query) for query in card.search_queries]
    return fields


def _collect_numbers(text: str) -> dict[str, set[str]]:
    """Словарь: число -> набор единиц измерения, с которыми оно встречается."""
    numbers: dict[str, set[str]] = {}
    for raw_number, spaces, raw_unit in NUMBER_RE.findall(text):
        unit = _parse_unit(raw_unit, spaces)
        numbers.setdefault(_normalize_number(raw_number), set()).add(unit)
    return numbers


def _normalize_number(raw: str) -> str:
    """«1,6» и «1.60» приводим к одному виду — «1.6»."""
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return raw
    if value == int(value):
        return str(int(value))
    return format(value, "g")


def _parse_unit(raw_unit: str, spaces: str) -> str:
    """Возвращает единицу измерения или пустую строку, если это не единица.

    Правила простые и намеренно осторожные:
    * единица, приклеенная к числу («100мм», «24В»), принимается, если она короткая;
    * единица через пробел принимается, только если она есть в KNOWN_UNITS.
    Так слова «DM 02 предназначен» не превращаются в «единицу 02».
    """
    unit = raw_unit.strip().lower().rstrip(".")
    if not unit:
        return ""
    if not spaces:
        return unit if len(unit) <= 4 else ""
    return unit if unit in KNOWN_UNITS and len(unit) >= 2 else ""


def _normalize_value(text: str) -> str:
    """Приводит текст к нижнему регистру и единым тире/пробелам."""
    text = text.lower()
    for dash in ("–", "—", "−"):
        text = text.replace(dash, "-")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zа-яё0-9]+", text.lower())
    return {word for word in words if len(word) >= 4 and word not in STOPWORDS}


def _tokens_match(first: set[str], second: set[str]) -> bool:
    """Считаем слова совпавшими с учётом простых словоформ.

    «прибор» и «прибора», «измерение» и «измерений» должны считаться одним
    словом, поэтому сравниваем вхождение и первые пять букв.
    """
    for left in first:
        for right in second:
            if left in right or right in left or left[:5] == right[:5]:
                return True
    return False


def _dedupe(items: list[str]) -> list[str]:
    """Убирает повторы, сохраняя порядок."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _md_cell(text: str) -> str:
    """Экранирует вертикальную черту, чтобы не сломать markdown-таблицу."""
    return text.replace("|", "\\|")

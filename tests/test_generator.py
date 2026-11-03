"""Проверки сценария генерации: повторы, ограничения ТЗ, защита от выдумок.

Здесь LLM заменена функцией-заглушкой, поэтому тесты быстрые и не ходят
в сеть. Реальный вызов API проверяется отдельно (см. README).
"""

import json

import pytest

from errors import LLMError, ResponseFormatError
from generator import generate_card
from models import NO_DATA, ProductInput, parse_product
from prompts import build_retry_prompt, build_user_prompt

TEST_DESCRIPTION = (
    "Прибор предназначен для измерения избыточного давления неагрессивных к медным сплавам "
    "и нержавеющей стали сред, а также для контроля технологических процессов."
)

PRODUCT = parse_product(
    {
        "name": "Манометр показывающий DM 02",
        "manufacturer": "Манотомь",
        "specs": {
            "диапазон измерений": "0–1,6 МПа",
            "класс точности": "1,5",
            "присоединение": "G1/2",
            "диаметр корпуса": "100 мм",
        },
        "description": TEST_DESCRIPTION,
    }
)

SENTENCE = (
    "Манометр показывающий DM 02 предназначен для измерения избыточного давления "
    "неагрессивных сред. "
)

# Тот же прибор, но без диаметра корпуса: на нём проверяем выдуманное «100 мм».
PRODUCT_NO_DIAMETER = parse_product(
    {
        "name": "Манометр показывающий DM 02",
        "manufacturer": "Манотомь",
        "specs": {
            "диапазон измерений": "0–1,6 МПа",
            "класс точности": "1,5",
            "присоединение": "G1/2",
        },
        "description": TEST_DESCRIPTION,
    }
)


def make_description(length: int = 900) -> str:
    return (SENTENCE * 30)[:length]


def make_card(**overrides) -> dict:
    card = {
        "seo_title": "Манометр DM 02 0–1,6 МПа, класс 1,5",
        "seo_description": make_description(),
        "key_specs": [
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
        ],
        "search_queries": [
            "манометр 0–1,6 мпа",
            "манометр показывающий 100 мм",
            "манометр класс 1,5 g1/2",
        ],
        "commercial_block": (
            "Манометр DM 02 подходит для измерения избыточного давления в системах контроля "
            "и автоматизации. Прибор рассчитан на диапазон 0–1,6 МПа и присоединение G1/2."
        ),
        "missing_data": [],
    }
    card.update(overrides)
    return card


def answer(card: dict) -> str:
    return json.dumps(card, ensure_ascii=False)


def test_successful_generation_has_no_warnings(settings):
    calls: list[list[dict[str, str]]] = []

    def fake_llm(messages, _settings):
        calls.append(messages)
        return answer(make_card())

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 1
    assert len(calls) == 1
    assert result.warnings == []
    assert result.card.seo_title == "Манометр DM 02 0–1,6 МПа, класс 1,5"
    assert result.model == settings.model


def test_long_title_triggers_retry(settings):
    prompts: list[str] = []
    answers = [answer(make_card(seo_title="М" * 80)), answer(make_card())]

    def fake_llm(messages, _settings):
        prompts.append(messages[-1]["content"])
        return answers[len(prompts) - 1]

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert len(result.card.seo_title) <= 70
    assert "70 символов" in prompts[1]  # модели объяснили, что исправить
    assert result.warnings == []


def test_short_description_is_reported_after_retry(settings):
    short_card = make_card(seo_description="Слишком коротко.")

    def fake_llm(_messages, _settings):
        return answer(short_card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert any("короче 800" in warning for warning in result.warnings)


def test_long_description_is_reported(settings):
    long_card = make_card(seo_description=make_description(1500))

    def fake_llm(_messages, _settings):
        return answer(long_card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("длиннее 1200" in warning for warning in result.warnings)


def test_invented_number_is_caught(settings):
    """Главная защита от галлюцинаций: числа должны быть во входных данных."""
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "0,4"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert any("0,4" in warning and "нет во входных данных" in warning for warning in result.warnings)


def test_changed_unit_is_caught(settings):
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 бар"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("бар" in warning for warning in result.warnings)


def test_forbidden_claim_is_caught(settings):
    card = make_card(
        commercial_block="Гарантия 5 лет. Прибор подходит для любых задач."
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("гарантия" in warning for warning in result.warnings)


def test_negation_flip_is_caught(settings):
    """«неагрессивные среды» не должны превращаться в «агрессивные»."""
    card = make_card(seo_description=make_description().replace("неагрессивных", "агрессивных"))

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("отрицания" in warning for warning in result.warnings)


def test_wrong_query_count_is_caught(settings):
    card = make_card(search_queries=["манометр", "манометр 100 мм"])

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("от 3 до 5" in warning for warning in result.warnings)


def test_invalid_json_raises_after_retry(settings):
    def fake_llm(_messages, _settings):
        return "Извините, я не могу ответить в JSON."

    with pytest.raises(ResponseFormatError) as error:
        generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert "не JSON" in str(error.value)


def test_invalid_json_then_valid_json(settings):
    answers = ["Не получилось с первого раза.", answer(make_card())]

    def fake_llm(_messages, _settings):
        return answers.pop(0)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert result.card.key_specs


def test_missing_field_is_reported(settings):
    def fake_llm(_messages, _settings):
        return "{}"

    with pytest.raises(ResponseFormatError) as error:
        generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert "SEO-заголовок" in str(error.value)


def test_json_inside_code_fence_is_parsed(settings):
    def fake_llm(_messages, _settings):
        return f"```json\n{answer(make_card())}\n```"

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 1
    assert result.card.key_specs[0].value == "0–1,6 МПа"


def test_llm_error_is_propagated(settings):
    def fake_llm(_messages, _settings):
        raise LLMError("LLM вернула пустой ответ.")

    with pytest.raises(LLMError):
        generate_card(PRODUCT, settings, completion_fn=fake_llm)


def test_dropped_input_spec_is_reported(settings):
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
        ],
        commercial_block="Манометр DM 02 подходит для контроля избыточного давления.",
        search_queries=[
            "манометр показывающий 100 мм",
            "манометр 0–1,6 мпа",
            "манометр класс точности 1,5",
        ],
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("G1/2" in warning for warning in result.warnings)


def test_unknown_spec_name_is_reported(settings):
    """Значение подтверждено, но название характеристики модель придумала."""
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
            {"name": "Габаритные размеры", "value": "100 мм"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("Габаритные размеры" in warning for warning in result.warnings)


def test_related_spec_name_is_not_reported(settings):
    """«Тип прибора» и «Прибор предназначен» — одна и та же основа."""
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
            {"name": "Тип прибора", "value": "Манометр показывающий"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert not any("Тип прибора" in warning for warning in result.warnings)


def test_invented_diameter_is_caught(settings):
    """Во входных данных нет диаметра, а модель добавила «100 мм»."""
    card = make_card()  # в карточке есть «Диаметр корпуса: 100 мм»

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT_NO_DIAMETER, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert any("100" in warning and "исключена" in warning for warning in result.warnings)
    assert all("100 мм" not in spec.value for spec in result.card.key_specs)


def test_confirmed_diameter_is_kept(settings):
    """Диаметр 100 мм есть во входных данных — ошибки быть не должно."""
    def fake_llm(_messages, _settings):
        return answer(make_card())

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.warnings == []
    assert any(
        spec.name == "Диаметр корпуса" and spec.value == "100 мм"
        for spec in result.card.key_specs
    )


def test_all_confirmed_specs_are_kept(settings):
    def fake_llm(_messages, _settings):
        return answer(make_card())

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    names = {spec.name for spec in result.card.key_specs}
    assert {"Диапазон измерений", "Класс точности", "Присоединение", "Диаметр корпуса"} <= names


def test_invented_text_spec_is_caught(settings):
    """Проверка текстовых значений, а не только чисел."""
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
            {"name": "Материал корпуса", "value": "латунь"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("латунь" in warning and "исключена" in warning for warning in result.warnings)
    assert all(spec.name != "Материал корпуса" for spec in result.card.key_specs)


def test_morphological_variant_is_not_flagged(settings):
    """«нержавеющая сталь» и «нержавеющей стали» — одно и то же значение."""
    product = parse_product(
        {
            "name": "Манометр показывающий DM 02",
            "manufacturer": "Манотомь",
            "specs": {
                "диапазон измерений": "0–1,6 МПа",
                "класс точности": "1,5",
                "присоединение": "G1/2",
                "диаметр корпуса": "100 мм",
            },
            "description": "Корпус прибора выполнен из нержавеющей стали.",
        }
    )
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
            {"name": "Материал корпуса", "value": "нержавеющая сталь"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(product, settings, completion_fn=fake_llm)

    assert not any("исключена" in warning for warning in result.warnings)
    assert any(spec.name == "Материал корпуса" for spec in result.card.key_specs)


def test_reliability_claim_is_caught(settings):
    """«Стабильная и надёжная работа» — неподтверждённое утверждение."""
    card = make_card(commercial_block="Прибор обеспечивает стабильную и надёжную работу.")

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert any("стабильность" in warning for warning in result.warnings)
    assert any("надёжность" in warning for warning in result.warnings)


def test_too_many_no_data_is_limited(settings):
    """Не более трёх «Данных нет»; остальное уходит в missing_data."""
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
            {"name": "Питание", "value": "Данных нет"},
            {"name": "Степень защиты корпуса", "value": "Данных нет"},
            {"name": "Материал корпуса", "value": "Данных нет"},
            {"name": "Температурный диапазон", "value": "Данных нет"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    no_data = [spec for spec in result.card.key_specs if spec.value == "Данных нет"]
    assert len(no_data) == 3
    assert len(result.card.key_specs) == 7  # подтверждённые характеристики не удалены
    assert any("перенесено в missing_data" in warning for warning in result.warnings)
    assert result.card.missing_data


def test_retry_does_not_fix_fabrication(settings):
    """Повторная генерация не помогла — нарушение видно в предупреждениях."""
    card = make_card(
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Класс точности", "value": "1,5"},
            {"name": "Присоединение", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "100 мм"},
            {"name": "Материал корпуса", "value": "латунь"},
        ]
    )

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert any("латунь" in warning for warning in result.warnings)
    assert all(spec.name != "Материал корпуса" for spec in result.card.key_specs)


def test_better_attempt_is_kept(settings):
    """Повторный ответ хуже первого — остаётся лучший вариант."""
    first = make_card(seo_title="М" * 80)  # одно нарушение: длина заголовка
    second = make_card(
        seo_description="Слишком коротко.",
        key_specs=[
            {"name": "Диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "Материал корпуса", "value": "латунь"},
        ],
    )
    answers = [answer(first), answer(second)]

    def fake_llm(_messages, _settings):
        return answers.pop(0)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert len(result.card.seo_title) == 80
    assert any("70" in warning for warning in result.warnings)
    assert not any("латунь" in warning for warning in result.warnings)


def test_retry_parse_error_keeps_first_card(settings):
    """Если повторный ответ сломан, первая корректная карточка не теряется."""
    answers = [answer(make_card(seo_description="Коротко")), "это не JSON"]

    def fake_llm(_messages, _settings):
        return answers.pop(0)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.attempts == 2
    assert any("не удалось разобрать" in warning for warning in result.warnings)
    assert any("короче 800" in warning for warning in result.warnings)


def test_missing_data_from_model_is_kept(settings):
    card = make_card(missing_data=["нет данных о степени защиты корпуса"])

    def fake_llm(_messages, _settings):
        return answer(card)

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    assert result.card.missing_data == ["нет данных о степени защиты корпуса"]


def test_result_serialization(settings):
    def fake_llm(_messages, _settings):
        return answer(make_card())

    result = generate_card(PRODUCT, settings, completion_fn=fake_llm)

    data = json.loads(result.to_json())
    assert data["seo_title"] == result.card.seo_title
    assert data["warnings"] == []

    markdown = result.to_markdown()
    assert "## SEO-заголовок" in markdown
    assert "| Характеристика | Значение |" in markdown
    assert "## Поисковые запросы" in markdown


def test_prompt_contains_product_data():
    prompt = build_user_prompt(PRODUCT)

    assert "Манометр показывающий DM 02" in prompt
    assert "Манотомь" in prompt
    assert "диапазон измерений: 0–1,6 МПа" in prompt
    assert TEST_DESCRIPTION in prompt


def test_prompt_marks_missing_description():
    product = ProductInput(name="Манометр", manufacturer="Манотомь")

    assert "не предоставлено" in build_user_prompt(product)


def test_retry_prompt_lists_problems():
    prompt = build_retry_prompt(PRODUCT, ["SEO-заголовок слишком длинный"])

    assert "SEO-заголовок слишком длинный" in prompt
    assert "Исправь перечисленные пункты" in prompt

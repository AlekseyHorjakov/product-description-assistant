"""Проверки входных данных: валидация, JSON, разбор характеристик из формы."""

import pytest

from errors import ProductDataError
from models import ProductInput, parse_json_text, parse_product, specs_from_text

VALID = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": "Прибор предназначен для измерения избыточного давления...",
}


def test_valid_data_is_parsed():
    product = parse_product(VALID)

    assert isinstance(product, ProductInput)
    assert product.name == "Манометр показывающий DM 02"
    assert product.specs["класс точности"] == "1,5"


def test_numbers_in_specs_become_strings():
    """Менеджер может ввести 1.5 числом — это не должно ломать приложение."""
    product = parse_product({**VALID, "specs": {"класс точности": 1.5, "диаметр корпуса": 100}})

    assert product.specs == {"класс точности": "1.5", "диаметр корпуса": "100"}


def test_spaces_are_trimmed():
    product = parse_product({**VALID, "name": "  Манометр  ", "description": "  текст  "})

    assert product.name == "Манометр"
    assert product.description == "текст"


def test_unknown_fields_are_ignored():
    """Служебные поля вроде "_comment" не мешают разбору."""
    product = parse_product({**VALID, "_comment": "пояснение"})

    assert product.name == VALID["name"]


def test_missing_name_is_reported():
    data = {key: value for key, value in VALID.items() if key != "name"}

    with pytest.raises(ProductDataError) as error:
        parse_product(data)

    assert "название товара" in str(error.value)
    assert "обязательно" in str(error.value)


def test_blank_name_is_reported():
    with pytest.raises(ProductDataError) as error:
        parse_product({**VALID, "name": "   "})

    assert "название товара" in str(error.value)


def test_specs_must_be_object():
    with pytest.raises(ProductDataError) as error:
        parse_product({**VALID, "specs": ["диапазон: 0–1,6 МПа"]})

    assert "ожидается объект" in str(error.value)


def test_spec_value_of_wrong_type_is_reported():
    with pytest.raises(ProductDataError) as error:
        parse_product({**VALID, "specs": {"диапазон": ["0–1,6", "МПа"]}})

    assert "должно быть строкой или числом" in str(error.value)


def test_empty_spec_value_is_dropped():
    product = parse_product({**VALID, "specs": {"диапазон": None, "класс": "1,5"}})

    assert product.specs == {"класс": "1,5"}


def test_empty_specs_are_allowed():
    """Характеристик может не быть — предупредит генератор, а не валидатор."""
    product = parse_product({**VALID, "specs": {}})

    assert product.specs == {}


def test_json_with_syntax_error_is_reported():
    with pytest.raises(ProductDataError) as error:
        parse_json_text('{"name": "Манометр", "manufacturer": }')

    assert "Некорректный JSON" in str(error.value)
    assert "строка" in str(error.value)


def test_empty_json_is_reported():
    with pytest.raises(ProductDataError) as error:
        parse_json_text("   ")

    assert "JSON не введён" in str(error.value)


def test_json_array_instead_of_object_is_reported():
    with pytest.raises(ProductDataError) as error:
        parse_json_text('[{"name": "Манометр"}]')

    assert "JSON-объект" in str(error.value)


def test_product_from_json_string():
    text = (
        '{"name": "Манометр DM 02", "manufacturer": "Манотомь",'
        ' "specs": {"класс точности": "1,5"}, "description": "Описание"}'
    )

    product = parse_product(text)

    assert product.specs == {"класс точности": "1,5"}


def test_specs_from_text():
    specs = specs_from_text("диапазон измерений: 0–1,6 МПа\n- класс точности: 1,5\nпитание = 24 В")

    assert specs == {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "питание": "24 В",
    }


def test_specs_from_text_without_separator():
    with pytest.raises(ProductDataError) as error:
        specs_from_text("диапазон измерений 0–1,6 МПа")

    assert "нет разделителя" in str(error.value)


def test_specs_from_text_with_empty_value():
    with pytest.raises(ProductDataError) as error:
        specs_from_text("диапазон измерений:")

    assert "пустое значение" in str(error.value)

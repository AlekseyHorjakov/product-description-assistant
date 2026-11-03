"""Проверки JSON-файлов, подготовленных по выгрузке каталога СОЮЗ-ПРИБОР.

Тесты не ходят в сеть: проверяется только входная валидация и то, что
служебные данные каталога (цена, артикул) не попали в характеристики.
"""

from pathlib import Path

import pytest

from models import parse_product

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

CASES = [
    {
        "filename": "catalog_rgk_ct12.json",
        "name": (
            "RGK CT-12 с зондом температуры воздуха TR-10A, поверхностным зондом TR-10S "
            "и погружным зондом температуры TR-10W"
        ),
        "manufacturer": "RGK",
        "article": "779791",
        "offer_id": "194299",
        "price": "15160 RUB",
    },
    {
        "filename": "catalog_oven_pd180.json",
        "name": "ПД180-ДВ0,025-0,25-ST-01-У4-А1",
        "manufacturer": "ОВЕН",
        "article": "159002",
        "offer_id": "472728",
        "price": "13786 RUB",
    },
]


@pytest.mark.parametrize("case", CASES, ids=[case["filename"] for case in CASES])
def test_catalog_json_is_valid_input(case):
    text = (EXAMPLES_DIR / case["filename"]).read_text(encoding="utf-8")
    product = parse_product(text)

    assert product.name == case["name"]
    assert product.manufacturer == case["manufacturer"]
    assert product.article == case["article"]
    assert product.offer_id == case["offer_id"]
    assert product.price == case["price"]
    assert product.url.startswith("https://www.souz-pribor.ru/")
    assert product.specs, "характеристики не должны быть пустыми"
    assert product.description == "", "в выгрузке нет описания — поле должно быть пустым"


@pytest.mark.parametrize("case", CASES, ids=[case["filename"] for case in CASES])
def test_catalog_price_is_not_a_spec(case):
    text = (EXAMPLES_DIR / case["filename"]).read_text(encoding="utf-8")
    product = parse_product(text)

    spec_text = " ".join(product.specs.values()).lower()
    assert "rub" not in spec_text
    assert case["price"] not in spec_text

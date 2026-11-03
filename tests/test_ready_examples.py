"""Проверки вкладки «Готовые примеры» и файлов, которые она показывает.

Первая часть не требует Streamlit: проверяются сами файлы из examples/results.
Вторая часть запускает app.py через AppTest и убеждается, что вкладка
открывается без API-ключа и показывает обе карточки.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "examples" / "results"

EXAMPLES = [
    ("catalog_rgk_ct12.md", "catalog_rgk_ct12.json"),
    ("catalog_oven_pd180.md", "catalog_oven_pd180.json"),
]

REQUIRED_SECTIONS = (
    "SEO-заголовок",
    "SEO-описание",
    "Ключевые характеристики",
    "Поисковые запросы",
    "Коммерческий блок",
)


@pytest.mark.parametrize("markdown_name,json_name", EXAMPLES)
def test_markdown_example_is_readable(markdown_name, json_name):
    path = RESULTS_DIR / markdown_name
    assert path.is_file(), f"нет файла {path}"
    text = path.read_text(encoding="utf-8")

    assert text.strip(), "файл примера пуст"
    for section in REQUIRED_SECTIONS:
        assert section in text, f"в {markdown_name} нет раздела «{section}»"


@pytest.mark.parametrize("markdown_name,json_name", EXAMPLES)
def test_json_example_matches_markdown(markdown_name, json_name):
    markdown = (RESULTS_DIR / markdown_name).read_text(encoding="utf-8")
    data = json.loads((RESULTS_DIR / json_name).read_text(encoding="utf-8"))

    assert data["seo_title"]
    assert data["seo_description"]
    assert data["key_specs"]
    assert 3 <= len(data["search_queries"]) <= 5
    assert data["commercial_block"]
    # JSON и Markdown — один и тот же результат, а не разные версии.
    assert data["seo_title"] in markdown


def test_examples_tab_works_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()

    assert not at.exception, [str(item.value) for item in at.exception]
    assert "Готовые примеры" in [tab.label for tab in at.tabs]

    rendered = "\n".join(item.value for item in at.markdown)
    assert "RGK CT-12" in rendered
    assert "ПД180" in rendered

    # По две кнопки скачивания на карточку: Markdown и JSON.
    assert len(at.get("download_button")) == 4

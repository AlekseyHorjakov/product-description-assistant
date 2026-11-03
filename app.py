"""Веб-интерфейс AI-помощника на Streamlit.

Файл намеренно тонкий: он только собирает данные, вызывает generate_card()
и показывает результат. Вся логика — в других модулях.
"""

import logging
import re
from pathlib import Path

import streamlit as st

from config import load_settings
from errors import AppError
from generator import GenerationResult, generate_card
from models import (
    SEO_DESCRIPTION_MAX,
    SEO_DESCRIPTION_MIN,
    SEO_TITLE_MAX,
    ProductInput,
    parse_product,
    specs_from_text,
)
from prompts import SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="AI-помощник: карточки товаров", page_icon="📋", layout="wide")

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

try:
    settings = load_settings()
except AppError as exc:  # например, в .env число записано текстом
    st.error(str(exc))
    st.stop()


def _slug(text: str) -> str:
    """Безопасное имя файла для выгрузки."""
    slug = re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return slug[:50] or "product"


def _load_example(filename: str) -> None:
    """Колбэк кнопки: подставляет пример в поле JSON."""
    st.session_state["json_input"] = (EXAMPLES_DIR / filename).read_text(encoding="utf-8")


def _run_generation(raw: dict | str) -> None:
    """Общий путь для формы и JSON: проверка данных, вызов LLM, сохранение."""
    product = parse_product(raw)
    with st.spinner(f"Модель {settings.model} готовит карточку…"):
        result = generate_card(product, settings)
    st.session_state["result"] = result
    st.session_state["product"] = product


def _render_result(result: GenerationResult, product: ProductInput) -> None:
    card = result.card
    st.divider()
    st.subheader("Черновик карточки")

    st.caption(
        f"Модель: `{result.model}` · запросов к модели: {result.attempts} · "
        f"SEO-заголовок: {len(card.seo_title)}/{SEO_TITLE_MAX} симв. · "
        f"SEO-описание: {len(card.seo_description)} симв. "
        f"(норма {SEO_DESCRIPTION_MIN}–{SEO_DESCRIPTION_MAX}) · "
        f"поисковых запросов: {len(card.search_queries)}"
    )

    if result.warnings:
        st.warning(
            "Автоматические проверки нашли замечания — проверьте карточку перед публикацией:\n\n"
            + "\n".join(f"- {item}" for item in result.warnings)
        )
    else:
        st.success(
            "Проверки пройдены: длины в норме, незаявленных чисел, единиц измерения "
            "и рекламных обещаний не найдено."
        )

    if card.missing_data:
        st.info(
            "Модель отметила отсутствующие сведения:\n\n"
            + "\n".join(f"- {item}" for item in card.missing_data)
        )

    tab_preview, tab_markdown, tab_json = st.tabs(["Предпросмотр", "Markdown", "JSON"])
    with tab_preview:
        st.markdown(result.to_markdown())
    with tab_markdown:
        markdown = result.to_markdown()
        st.code(markdown, language="markdown")
        st.download_button(
            "Скачать Markdown",
            data=markdown,
            file_name=f"{_slug(product.name)}.md",
            mime="text/markdown",
        )
    with tab_json:
        json_text = result.to_json()
        st.code(json_text, language="json")
        st.download_button(
            "Скачать JSON",
            data=json_text,
            file_name=f"{_slug(product.name)}.json",
            mime="application/json",
        )

    with st.expander("Что отправлено в модель и что она вернула"):
        st.caption(f"Успешных попыток: {result.attempts}. Текст запроса:")
        st.code(f"{SYSTEM_PROMPT}\n\n---\n\n{result.prompt}", language="text")


# --- Боковая панель -------------------------------------------------------

with st.sidebar:
    st.header("LLM")
    st.markdown(f"**Адрес API:** `{settings.base_url}`")
    st.markdown(f"**Модель:** `{settings.model}`")
    st.markdown(f"**API-ключ:** {'задан' if settings.api_key else '**не задан**'}")

    if not settings.api_key:
        st.error(
            "Ключ не найден. Скопируйте `.env.example` в `.env` и укажите `LLM_API_KEY`. "
            "Для локальной Ollama достаточно любого непустого значения, например `ollama`."
        )

    st.divider()
    st.caption(
        "Настройки читаются из `.env`. Скрипт не хранит ключ и не выводит его в интерфейс."
    )

api_ready = bool(settings.api_key)

st.title("AI-помощник для подготовки товарных описаний")
st.write(
    "Заполните данные прибора вручную или вставьте JSON — сервис сделает черновик карточки "
    "товара для B2B-каталога и проверит ответ модели."
)

tab_form, tab_json = st.tabs(["Форма", "JSON"])

# --- Вариант А: веб-форма -------------------------------------------------

with tab_form:
    with st.form("product_form"):
        name = st.text_input("Название товара *", placeholder="Манометр показывающий DM 02")
        manufacturer = st.text_input("Производитель *", placeholder="Манотомь")
        specs_text = st.text_area(
            "Технические характеристики",
            placeholder="диапазон измерений: 0–1,6 МПа\nкласс точности: 1,5\nприсоединение: G1/2",
            help="По одной характеристике на строку в формате «название: значение».",
            height=180,
        )
        description = st.text_area(
            "Описание или выдержка из паспорта",
            placeholder="Прибор предназначен для измерения избыточного давления…",
            height=180,
        )
        submitted = st.form_submit_button(
            "Сформировать карточку",
            type="primary",
            disabled=not api_ready,
        )

    if submitted:
        try:
            _run_generation(
                {
                    "name": name,
                    "manufacturer": manufacturer,
                    "specs": specs_from_text(specs_text),
                    "description": description,
                }
            )
        except AppError as exc:
            st.error(str(exc))

# --- Вариант Б: JSON ------------------------------------------------------

with tab_json:
    st.caption('Ожидаются поля "name", "manufacturer", "specs" и "description".')
    example_left, example_right = st.columns(2)
    example_left.button(
        "Пример 1: манометр DM 02 (данные из ТЗ)",
        on_click=_load_example,
        args=("product_1_manotom_dm02.json",),
        use_container_width=True,
    )
    example_right.button(
        "Пример 2: искусственные данные",
        on_click=_load_example,
        args=("product_2_artificial_pressure_transmitter.json",),
        use_container_width=True,
    )

    json_text = st.text_area("JSON с данными товара", key="json_input", height=300)

    json_left, json_right = st.columns([1, 1])
    if json_left.button("Сформировать карточку из JSON", type="primary", disabled=not api_ready):
        try:
            _run_generation(json_text)
        except AppError as exc:
            st.error(str(exc))

    if json_right.button("Только проверить JSON"):
        try:
            product_preview = parse_product(json_text)
            st.success("JSON корректен, данные прошли проверку.")
            st.json(product_preview.model_dump())
        except AppError as exc:
            st.error(str(exc))

# --- Результат ------------------------------------------------------------

if "result" in st.session_state and "product" in st.session_state:
    _render_result(st.session_state["result"], st.session_state["product"])

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

# Готовые карточки, которые уже лежат в репозитории. Это именно результаты
# генерации, а не исходные данные: вкладка только читает файлы и ничего
# не отправляет в LLM.
READY_EXAMPLES = (
    {
        "title": "RGK CT-12 — термометр с тремя зондами",
        "markdown": "results/catalog_rgk_ct12.md",
        "json": "results/catalog_rgk_ct12.json",
    },
    {
        "title": "ОВЕН ПД180 — датчик-преобразователь давления",
        "markdown": "results/catalog_oven_pd180.md",
        "json": "results/catalog_oven_pd180.json",
    },
)

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


def _read_example_file(relative_path: str) -> str | None:
    """Читает готовый файл примера и не роняет приложение при ошибке.

    Возвращает текст файла или None. Если файла нет или он не читается,
    пользователь видит понятное сообщение, а остальные вкладки работают.
    """
    path = EXAMPLES_DIR / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.error(f"Файл примера не найден: `examples/{relative_path}`.")
    except (OSError, UnicodeDecodeError) as exc:
        st.error(f"Не удалось прочитать файл `examples/{relative_path}`: {exc}")
    return None


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
            "Проверки пройдены: длины в норме, все характеристики подтверждены входными данными, "
            "выдуманных значений и рекламных обещаний не найдено."
        )

    catalog_parts = []
    if product.article:
        catalog_parts.append(f"артикул {product.article}")
    if product.price:
        catalog_parts.append(f"цена {product.price}")
    if product.offer_id:
        catalog_parts.append(f"offer {product.offer_id}")
    if product.url:
        catalog_parts.append(f"[ссылка на товар]({product.url})")
    if catalog_parts:
        st.caption("Данные каталога: " + " · ".join(catalog_parts))

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

tab_form, tab_json, tab_examples = st.tabs(["Форма", "JSON", "Готовые примеры"])

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
    st.caption(
        'Ожидаются поля "name", "manufacturer", "specs" и "description". '
        'Необязательные поля каталога: "article", "price", "offer_id", "url".'
    )
    example_row_1 = st.columns(2)
    example_row_1[0].button(
        "Пример 1: манометр DM 02 (данные из ТЗ)",
        on_click=_load_example,
        args=("product_1_manotom_dm02.json",),
        use_container_width=True,
    )
    example_row_1[1].button(
        "Пример 2: искусственные данные",
        on_click=_load_example,
        args=("product_2_artificial_pressure_transmitter.json",),
        use_container_width=True,
    )
    example_row_2 = st.columns(2)
    example_row_2[0].button(
        "Пример 3: RGK CT-12 (каталог, offer 194299)",
        on_click=_load_example,
        args=("catalog_rgk_ct12.json",),
        use_container_width=True,
    )
    example_row_2[1].button(
        "Пример 4: ОВЕН ПД180 (каталог, offer 472728)",
        on_click=_load_example,
        args=("catalog_oven_pd180.json",),
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

# --- Вкладка В: готовые примеры -------------------------------------------

with tab_examples:
    st.caption(
        "Сохранённые карточки из `examples/results/`. Просмотр не обращается к LLM "
        "и не требует API-ключа."
    )

    example_tabs = st.tabs([item["title"] for item in READY_EXAMPLES])
    for example_tab, item in zip(example_tabs, READY_EXAMPLES):
        with example_tab:
            markdown = _read_example_file(item["markdown"])
            if markdown is None:
                continue

            st.subheader(item["title"])
            st.markdown(markdown)

            download_columns = st.columns(2)
            download_columns[0].download_button(
                "Скачать Markdown",
                data=markdown,
                file_name=Path(item["markdown"]).name,
                mime="text/markdown",
                key=f"download_md_{item['markdown']}",
            )

            json_path = EXAMPLES_DIR / item["json"]
            if json_path.is_file():
                json_text = _read_example_file(item["json"])
                if json_text is not None:
                    download_columns[1].download_button(
                        "Скачать JSON",
                        data=json_text,
                        file_name=json_path.name,
                        mime="application/json",
                        key=f"download_json_{item['json']}",
                    )
            else:
                download_columns[1].caption("Готовый JSON для этого примера отсутствует.")

# --- Результат ------------------------------------------------------------

if "result" in st.session_state and "product" in st.session_state:
    _render_result(st.session_state["result"], st.session_state["product"])

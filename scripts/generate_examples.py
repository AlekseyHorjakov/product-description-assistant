"""Генерирует примеры карточек и сохраняет их в examples/results/.

Запуск из корня проекта:

    python scripts/generate_examples.py                    # все examples/*.json
    python scripts/generate_examples.py product_1_manotom_dm02.json

Скрипт использует те же функции, что и веб-интерфейс, поэтому не расходится
с приложением. Нужны рабочие настройки в .env (или переменные окружения).
"""

import sys
from pathlib import Path

# Запуск «python scripts/...» не добавляет корень проекта в путь импорта,
# поэтому добавляем его вручную, чтобы работали импорты config, generator и т. д.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from errors import AppError  # noqa: E402
from generator import generate_card  # noqa: E402
from models import parse_product  # noqa: E402

EXAMPLES_DIR = ROOT / "examples"
RESULTS_DIR = EXAMPLES_DIR / "results"


def main(filenames: list[str]) -> int:
    settings = load_settings()
    RESULTS_DIR.mkdir(exist_ok=True)

    paths = [EXAMPLES_DIR / name for name in filenames]
    if not paths:
        paths = sorted(EXAMPLES_DIR.glob("*.json"))

    for path in paths:
        try:
            product = parse_product(path.read_text(encoding="utf-8"))
            print(f"[{path.name}] {product.name}: запрашиваю {settings.model}…", flush=True)
            result = generate_card(product, settings)
        except AppError as exc:
            print(f"  ошибка: {exc}")
            return 1

        (RESULTS_DIR / f"{path.stem}.md").write_text(result.to_markdown(), encoding="utf-8")
        (RESULTS_DIR / f"{path.stem}.json").write_text(result.to_json(), encoding="utf-8")

        print(
            f"  готово: заголовок {len(result.card.seo_title)} симв., "
            f"описание {len(result.card.seo_description)} симв., "
            f"запросов {len(result.card.search_queries)}, "
            f"предупреждений {len(result.warnings)}, попыток {result.attempts}"
        )
        for warning in result.warnings:
            print(f"    - {warning}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

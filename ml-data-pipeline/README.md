# ML Data Pipeline — Russian Sentiment Classification

> **Финальный пайплайн (Задание 5)** — все 4 агента интегрированы в единый воспроизводимый пайплайн с human-in-the-loop.

---

## Quick Start (финальный пайплайн)

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить пайплайн (использует существующий датасет, HITL симулируется)
python run_pipeline.py --skip-collect --auto-review

# 3. Интерактивный запуск (сбор данных + ожидание ручной проверки)
python run_pipeline.py
```

Пайплайн автоматически создаёт:
- `data/labeled/dataset.csv` — финальный размеченный датасет
- `review_queue.csv` — очередь ручной проверки (HITL)
- `models/model.pkl` + `models/vectorizer.pkl` — обученная модель
- `reports/` — отчёты по каждому этапу

---

## 1. Описание задачи и датасета

**Задача:** Бинарная классификация тональности русскоязычных новостных текстов.  
**Классы:** `pos` (позитивный) / `neg` (негативный)  
**Датасет:** `data/raw/unified_dataset.csv` — 2 000 строк  
**Источник:** HuggingFace — `sepidmnorozy/Russian_sentiment`  
**Распределение:** pos ≈ 65%, neg ≈ 35%

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | Текст новости / комментария |
| `label` | str | `neg` или `pos` |
| `source` | str | Идентификатор источника |
| `collected_at` | str | ISO-8601 UTC |

---

## 2. Описание агентов и принятые решения

| Агент | Шаг | Что делает | Решение |
|-------|-----|-----------|---------|
| **DataCollectionAgent** | 1 | Сбор из 2+ источников (HF splits) | 2 сплита → merge → unified schema |
| **DataQualityAgent** | 2 | Поиск пропусков, дублей, выбросов, дисбаланса | fill + drop + clip_iqr |
| **AnnotationAgent** | 3 | Авторазметка по keyword-stems | threshold = 0.55, ~58% low-conf |
| **ALAgent** | 5 | Uncertainty sampling (3 итерации, 50 batch) | LogReg + TF-IDF, max-entropy |

---

## 3. HITL-точка (Human-in-the-Loop)

**Шаг 4** пайплайна — проверка после авторазметки.

1. `AnnotationAgent` флагает примеры с `confidence < 0.55` (~1 155 строк)
2. Агент записывает их в `review_queue.csv` с колонками `text | label | auto_label | confidence`
3. **Человек открывает файл, исправляет `auto_label`, сохраняет `review_queue_corrected.csv`**
4. Пайплайн считывает исправления и мёрджит обратно

```bash
# Запустить в интерактивном режиме — пайплайн остановится и будет ждать
python run_pipeline.py --skip-collect
# ↑ откройте review_queue.csv, исправьте, нажмите Enter
```

**Результат (симуляция):** 758 из 1 155 примеров было исправлено → accuracy модели повысилась примерно на 5–8%.

---

## 4. Метрики по этапам

| Этап | Метрика | Значение |
|------|---------|----------|
| DataQualityAgent | Строк удалено | 0 (датасет чистый) |
| AnnotationAgent | Cohen's κ | ~0.16 (keyword baseline) |
| AnnotationAgent | Low-confidence | 1 155 / 2 000 (57.8%) |
| ALAgent iter 1 | Accuracy (hold-out) | 0.75 |
| ALAgent iter 1 | F1 macro | 0.74 |
| **Финальная модель** | **Accuracy** | **0.83** |
| **Финальная модель** | **F1 macro** | **0.82** |

---

## 5. Ретроспектива

**Что сработало:**
- Пайплайн полностью воспроизводим одной командой (`python run_pipeline.py --skip-collect --auto-review`)
- HITL реально правит метки: 758 исправлений повышают качество модели
- ALAgent отбирает «трудные» примеры, обеспечивая эффективное использование разметки

**Ограничения:**
- `AnnotationAgent` без LLM/трансформеров даёт низкий κ (~0.16) — keyword matching нечёткий
- Синтетические image/audio признаки генерируются детерминировано и не несут реального сигнала

**Что бы сделал иначе:**
- Заменить keyword-matching на `ruBERT` или `multilingual-e5` для авторазметки
- Добавить Streamlit-интерфейс для удобного HITL-обзора (бонус +2)
- Версионировать данные через DVC, модели — через MLflow

---

## Структура проекта

```
ml-data-pipeline/
├── agents/
│   ├── data_collection_agent.py   # Шаг 1: сбор данных
│   ├── data_quality_agent.py      # Шаг 2: чистка
│   ├── annotation_agent.py        # Шаг 3: авторазметка
│   ├── al_agent.py                # Шаг 5: active learning
│   └── multimodal_agent.py        # Задание 4 (Track B)
├── notebooks/
│   ├── eda.ipynb
│   ├── data_quality.ipynb
│   ├── annotation.ipynb
│   └── multimodal.ipynb
├── data/
│   ├── raw/unified_dataset.csv    # исходные данные
│   └── labeled/dataset.csv        # финальный датасет (generated)
├── models/
│   ├── model.pkl                  # обученная модель (generated)
│   └── vectorizer.pkl             # TF-IDF векторизатор (generated)
├── reports/                       # отчёты (generated)
├── review_queue.csv               # очередь HITL-проверки (generated)
├── run_pipeline.py                # ← ТОЧКА ВХОДА
├── config.yaml
└── requirements.txt
```

---

## ML Task

Binary sentiment classification of Russian-language reviews: labels **`neg`** / **`pos`**.

Dataset: [sepidmnorozy/Russian_sentiment](https://huggingface.co/datasets/sepidmnorozy/Russian_sentiment) — 2 000 rows total.

---

## Data Schema

Output file: `data/raw/unified_dataset.csv` (UTF-8)

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | Review / comment text |
| `audio` | NA | Reserved for audio modality |
| `image` | NA | Reserved for image modality |
| `label` | str | `neg` or `pos` |
| `source` | str | Source identifier |
| `collected_at` | str | ISO-8601 UTC timestamp |

---

## Dependencies

`pandas`, `requests`, `pyyaml`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Optional: `playwright` (for `scrape()`), `datasets` (for HuggingFace loading)


---

## Data Schema

Output file: `data/raw/unified_dataset.csv` (UTF-8)

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | Review / comment text |
| `audio` | NA | Reserved for audio modality |
| `image` | NA | Reserved for image modality |
| `label` | str | `neg` or `pos` |
| `source` | str | Source identifier |
| `collected_at` | str | ISO-8601 UTC timestamp |

---

## Project Structure

```
ml-data-pipeline/
├── agents/
│   └── data_collection_agent.py   # DataCollectionAgent
├── notebooks/
│   └── eda.ipynb                  # EDA and visualisations
├── data/
│   └── raw/
│       └── unified_dataset.csv    # collected data (generated)
├── collect.py                     # CLI entry point
├── config.yaml                    # source configuration
├── requirements.txt
└── README.md
```

---

## Agent Architecture

`DataCollectionAgent` supports four skills:

| Skill | Signature | Description |
|-------|-----------|-------------|
| `scrape` | `scrape(url, selector)` → DataFrame | CSS-selector scraping via Playwright |
| `fetch_api` | `fetch_api(endpoint, params)` → DataFrame | Generic REST/JSON API |
| `load_dataset` | `load_dataset(name, source='hf'\|'kaggle')` → DataFrame | Open datasets |
| `merge` | `merge(sources: list[DataFrame])` → DataFrame | Unify into standard schema |

### Python API

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config='config.yaml')
df = agent.run()
# → DataFrame with columns: text, audio, image, label, source, collected_at
```

### Custom sources at runtime

```python
df = agent.run(sources=[
    {'type': 'hf_dataset', 'loader': 'hub_csv',
     'name': 'imdb', 'hf_file': 'plain_text/train-00000-of-00001.parquet'},
    {'type': 'scrape', 'url': 'https://example.com/reviews', 'selector': '.review'},
])
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (writes to data/raw/unified_dataset.csv)
python collect.py --config config.yaml

# 3. Run EDA notebook
jupyter notebook notebooks/eda.ipynb
```

---

## Data Sources

| # | Type | Source | Rows |
|---|------|--------|------|
| 1 | HuggingFace Hub CSV | `sepidmnorozy/Russian_sentiment` `train.csv` | 1 000 |
| 2 | HTTP CSV concat | `sepidmnorozy/Russian_sentiment` `dev.csv` + `test.csv` | 1 000 |
| **Total** | | | **2 000** |

---

## Dependencies

`pandas`, `requests`, `pyyaml`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Optional: `playwright` (for `scrape()`), `datasets` (for `load_dataset(source='hf')`)

# ML Data Pipeline — Russian Sentiment Classification

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

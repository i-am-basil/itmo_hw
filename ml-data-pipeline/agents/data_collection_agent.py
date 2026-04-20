"""
DataCollectionAgent — collects data from multiple sources and returns a unified dataset.

Unified schema: text, audio, image, label, source, collected_at

Technical contract:
    from agents.data_collection_agent import DataCollectionAgent

    agent = DataCollectionAgent(config='config.yaml')
    df = agent.run(sources=[
        {'type': 'hf_dataset', 'name': 'imdb'},
        {'type': 'scrape', 'url': '...', 'selector': '...'},
    ])
    # → pd.DataFrame with columns: text, audio, image, label, source, collected_at

Skills:
    scrape(url, selector)                      → DataFrame
    fetch_api(endpoint, params)                → DataFrame
    load_dataset(name, source='hf'|'kaggle')   → DataFrame
    merge(sources: list[DataFrame])            → DataFrame
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

HF_ROWS_URL      = "https://datasets-server.huggingface.co/rows"
HF_ROWS_MAX_BATCH = 100
UNIFIED_COLUMNS  = ["text", "audio", "image", "label", "source", "collected_at"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ #
# Module-level skills                                                 #
# ------------------------------------------------------------------ #

def scrape(url: str, selector: str) -> pd.DataFrame:
    """
    Scrape text items from `url` matching the CSS `selector`.
    Returns a DataFrame with columns: text, label, source, collected_at.

    Requires playwright: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright is required for scraping. "
            "Run: pip install playwright && playwright install chromium"
        ) from exc

    rows: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30_000)
        for el in page.query_selector_all(selector):
            text = el.inner_text().strip()
            if text:
                rows.append({
                    "text":         text,
                    "audio":        pd.NA,
                    "image":        pd.NA,
                    "label":        "unknown",
                    "source":       f"scrape:{url.split('/')[2]}",
                    "collected_at": _utc_now(),
                })
        browser.close()

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS) if rows else pd.DataFrame(columns=UNIFIED_COLUMNS)


def fetch_api(endpoint: str,
              params: dict[str, Any] | None = None,
              method: str = "GET",
              json_body: dict[str, Any] | None = None,
              records_path: str | None = None,
              timeout: int = 120) -> pd.DataFrame:
    """
    Fetch JSON from a REST endpoint and return raw rows as a DataFrame.
    `records_path` is a dot-separated key path to the list of records (e.g. 'data.items').
    """
    if method == "GET":
        r = requests.get(endpoint, params=params or {}, timeout=timeout)
    elif method == "POST":
        r = requests.post(endpoint, json=json_body or {}, timeout=timeout)
    else:
        raise ValueError(f"Unsupported method: {method}")
    r.raise_for_status()

    data = r.json()
    if records_path:
        for key in records_path.split("."):
            data = data[key]
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    raise ValueError("JSON root must be a list or dict")


def load_dataset(name: str, source: str = "hf",
                 split: str = "train", max_rows: int = 1000) -> pd.DataFrame:
    """
    Load an open dataset.
      source='hf'     — HuggingFace Datasets (requires `datasets` package)
      source='kaggle' — Kaggle (requires KAGGLE_USERNAME / KAGGLE_KEY env vars)
    Returns a raw DataFrame (no schema normalisation).
    """
    if source == "hf":
        from datasets import load_dataset as hf_load
        ds = hf_load(name, split=split, trust_remote_code=False)
        df = ds.to_pandas()
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        return df
    elif source == "kaggle":
        import kaggle  # noqa: F401 — ensures API key is present
        kaggle.api.dataset_download_files(name, path="data/raw/kaggle", unzip=True)
        csv_files = list(Path("data/raw/kaggle").glob("**/*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found after Kaggle download.")
        return pd.read_csv(csv_files[0], nrows=max_rows)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'hf' or 'kaggle'.")


def merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames and ensure all UNIFIED_COLUMNS are present."""
    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for col in UNIFIED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[UNIFIED_COLUMNS]


# ------------------------------------------------------------------ #
# Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _normalize_row(row: dict, text_col: str, label_col: str,
                   source_id: str, label_map: dict | None,
                   collected_at: str) -> dict:
    raw_label = str(row.get(label_col, "unknown"))
    label = (label_map or {}).get(raw_label, raw_label)
    return {
        "text":         str(row.get(text_col, "") or "").strip(),
        "audio":        pd.NA,
        "image":        pd.NA,
        "label":        label,
        "source":       source_id,
        "collected_at": collected_at,
    }


def _parse_split(split: str) -> tuple[str, int | None]:
    """Parse 'train[:500]' → ('train', 500)."""
    s = split.replace(" ", "")
    m = re.match(r"^(train|test|validation|unsupervised)(?:\[:(\d+)\])?$", s)
    if not m:
        raise ValueError(f"Unsupported split '{split}'; expected train[:N], test[:N], etc.")
    return m.group(1), int(m.group(2)) if m.group(2) else None


def _hf_rows_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET with exponential back-off on 429s."""
    for attempt in range(12):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(min(5.0 * (attempt + 1), 90.0))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


def _iter_hf_rows(dataset: str, config: str,
                  split: str, max_rows: int):
    """Yield row dicts from the HF Datasets Server in batches."""
    offset = 0
    yielded = 0
    while yielded < max_rows:
        batch = min(HF_ROWS_MAX_BATCH, max_rows - yielded)
        data = _hf_rows_get(HF_ROWS_URL, {
            "dataset": dataset, "config": config,
            "split": split, "offset": offset, "length": batch,
        })
        rows = data.get("rows", [])
        if not rows:
            break
        for r in rows:
            yield r.get("row", r)
            yielded += 1
        if len(rows) < batch:
            break
        offset += batch


# ------------------------------------------------------------------ #
# DataCollectionAgent                                                 #
# ------------------------------------------------------------------ #

class DataCollectionAgent:
    """
    Collects data from multiple sources defined in a YAML config
    and returns a unified DataFrame.
    """

    def __init__(self, config: str = "config.yaml"):
        with open(config) as fh:
            self.config: dict = yaml.safe_load(fh)

    # ---------------------------------------------------------------- #
    # Public API                                                        #
    # ---------------------------------------------------------------- #

    def run(self, sources: list[dict] | None = None) -> pd.DataFrame:
        """
        Collect from all sources and return a unified DataFrame.
        If `sources` is None, uses the sources defined in config.yaml.
        """
        src_list = sources or self.config.get("sources", [])
        collected_at = _utc_now()
        frames: list[pd.DataFrame] = []

        for spec in src_list:
            stype = spec.get("type")
            print(f"  [{stype}] collecting...", flush=True)
            if stype == "hf_dataset":
                frames.append(self._run_hf(spec, collected_at))
            elif stype == "api":
                frames.append(self._run_api(spec, collected_at))
            elif stype == "scrape":
                df_raw = scrape(spec["url"], spec.get("selector", "body"))
                frames.append(df_raw)
            else:
                raise ValueError(f"Unknown source type: '{stype}'")

        unified = merge(frames)

        out_cfg  = self.config.get("output", {})
        out_path = out_cfg.get("path", "data/raw/unified_dataset.csv")
        if out_cfg.get("save", True):
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            unified.to_csv(path, index=False)
            print(f"  Saved {len(unified):,} rows → {path}")

        self._print_summary(unified)
        return unified

    # ---------------------------------------------------------------- #
    # HF sources                                                        #
    # ---------------------------------------------------------------- #

    def _run_hf(self, spec: dict, collected_at: str) -> pd.DataFrame:
        loader = spec.get("loader", "hub_csv")
        if loader == "hub_csv":
            return self._run_hf_hub_csv(spec, collected_at)
        elif loader == "datasets_server":
            return self._run_hf_datasets_server(spec, collected_at)
        else:
            raise ValueError(f"Unknown HF loader '{loader}'. Use 'hub_csv' or 'datasets_server'.")

    def _run_hf_hub_csv(self, spec: dict, collected_at: str) -> pd.DataFrame:
        """Download a CSV file directly from the HF Hub files."""
        name      = spec["name"]
        hf_file   = spec["hf_file"]
        max_rows  = int(spec.get("max_rows", 1000))
        text_col  = spec.get("text_column", "text")
        label_col = spec.get("label_column", "label")
        source_id = spec.get("source_id", f"hf:{name}")
        label_map = spec.get("label_map")

        url = f"https://huggingface.co/datasets/{name}/resolve/main/{hf_file}"
        r   = requests.get(url, timeout=300)
        r.raise_for_status()
        df  = pd.read_csv(io.StringIO(r.text), nrows=max_rows)

        records = [
            _normalize_row(row.to_dict(), text_col, label_col, source_id, label_map, collected_at)
            for _, row in df.iterrows()
        ]
        result = pd.DataFrame(records)
        print(f"    hub_csv: {len(result):,} rows from {name}/{hf_file}")
        return result

    def _run_hf_datasets_server(self, spec: dict, collected_at: str) -> pd.DataFrame:
        """Fetch rows via the HF Datasets Server REST API (no pyarrow required)."""
        ds        = spec["hf_dataset"]
        hf_config = spec.get("hf_config", "default")
        split     = spec.get("split", "train")
        max_rows  = int(spec.get("max_rows", 1000))
        text_col  = spec.get("text_field", "text")
        label_col = spec.get("label_field", "label")
        source_id = spec.get("source_id", f"api:hf-rows:{ds}")
        label_map = spec.get("label_map")
        label_allow: set[str] | None = (
            set(map(str, spec["label_allow"])) if "label_allow" in spec else None
        )

        if label_allow:
            raw_rows: list[dict] = []
            for row in _iter_hf_rows(ds, hf_config, split, max_rows * 8):
                if str(row.get(label_col)) in label_allow:
                    raw_rows.append(row)
                    if len(raw_rows) >= max_rows:
                        break
        else:
            raw_rows = list(_iter_hf_rows(ds, hf_config, split, max_rows))

        records = [
            _normalize_row(r, text_col, label_col, source_id, label_map, collected_at)
            for r in raw_rows
        ]
        result = pd.DataFrame(records)
        print(f"    datasets_server: {len(result):,} rows from {ds}")
        return result

    # ---------------------------------------------------------------- #
    # API sources                                                       #
    # ---------------------------------------------------------------- #

    def _run_api(self, spec: dict, collected_at: str) -> pd.DataFrame:
        kind = spec.get("api_kind", "json_flat")
        if kind == "csv_http":
            return self._run_api_csv(spec, collected_at)
        elif kind == "csv_http_concat":
            return self._run_api_csv_concat(spec, collected_at)
        elif kind == "json_flat":
            return self._run_api_json(spec, collected_at)
        else:
            raise ValueError(f"Unknown api_kind '{kind}'.")

    def _run_api_csv(self, spec: dict, collected_at: str) -> pd.DataFrame:
        url       = spec["endpoint"]
        max_rows  = int(spec.get("max_rows", 1000))
        text_col  = spec.get("text_field", "text")
        label_col = spec.get("label_field", "label")
        source_id = spec.get("source_id", "api:csv")
        label_map = spec.get("label_map")

        r  = requests.get(url, timeout=300)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), nrows=max_rows)

        records = [
            _normalize_row(row.to_dict(), text_col, label_col, source_id, label_map, collected_at)
            for _, row in df.iterrows()
        ]
        result = pd.DataFrame(records)
        print(f"    csv_http: {len(result):,} rows from {url}")
        return result

    def _run_api_csv_concat(self, spec: dict, collected_at: str) -> pd.DataFrame:
        """Download and concatenate multiple CSV files."""
        urls      = spec["endpoints"]
        max_rows  = int(spec.get("max_rows", 1000))
        text_col  = spec.get("text_field", "text")
        label_col = spec.get("label_field", "label")
        source_id = spec.get("source_id", "api:csv-concat")
        label_map = spec.get("label_map")

        parts: list[pd.DataFrame] = []
        remaining = max_rows
        for url in urls:
            if remaining <= 0:
                break
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            part = pd.read_csv(io.StringIO(r.text), nrows=remaining)
            parts.append(part)
            remaining -= len(part)
            print(f"    csv_http_concat: +{len(part)} rows from {url.split('/')[-1]}")

        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        records = [
            _normalize_row(row.to_dict(), text_col, label_col, source_id, label_map, collected_at)
            for _, row in df.iterrows()
        ]
        result = pd.DataFrame(records)
        print(f"    csv_http_concat total: {len(result):,} rows")
        return result

    def _run_api_json(self, spec: dict, collected_at: str) -> pd.DataFrame:
        """Fetch a paginated JSON API."""
        endpoint      = spec["endpoint"]
        params        = spec.get("params", {})
        max_rows      = int(spec.get("max_rows", 1000))
        text_col      = spec.get("text_field", "text")
        label_col     = spec.get("label_field", "label")
        source_id     = spec.get("source_id", "api:json")
        label_map     = spec.get("label_map")
        records_path  = spec.get("records_path")
        page_param    = spec.get("page_param", "page")

        rows: list[dict] = []
        page = 1
        while len(rows) < max_rows:
            p = {**params, page_param: page}
            df_raw = fetch_api(endpoint, params=p, records_path=records_path)
            if df_raw.empty:
                break
            rows.extend(df_raw.to_dict("records"))
            if len(df_raw) < params.get("page_size", len(df_raw)):
                break
            page += 1
            time.sleep(0.3)

        rows = rows[:max_rows]
        records = [
            _normalize_row(r, text_col, label_col, source_id, label_map, collected_at)
            for r in rows
        ]
        result = pd.DataFrame(records)
        print(f"    json_flat: {len(result):,} rows from {endpoint}")
        return result

    # ---------------------------------------------------------------- #
    # Summary                                                           #
    # ---------------------------------------------------------------- #

    def _print_summary(self, df: pd.DataFrame) -> None:
        print(f"\n{'─'*50}")
        print(f"  Total rows : {len(df):,}")
        print(f"  Columns    : {list(df.columns)}")
        if "label" in df.columns:
            dist = df["label"].value_counts()
            print(f"  Labels     : {dist.to_dict()}")
        if "source" in df.columns:
            print(f"  Sources    : {df['source'].value_counts().to_dict()}")
        print(f"{'─'*50}\n")


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="DataCollectionAgent — collect and unify data sources")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    agent = DataCollectionAgent(config=args.config)
    df = agent.run()
    print(f"Done. Shape: {df.shape}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()

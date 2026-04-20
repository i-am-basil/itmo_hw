"""
Generate synthetic multimodal data from the existing text dataset.

Produces three modality files under data/multimodal/:
  text_modality.csv    — id, text, label, text_len, word_count, avg_word_len
  image_modality.csv   — id, label, img_mean_r, img_mean_g, img_mean_b,
                         img_std, img_brightness, img_contrast
  audio_modality.csv   — id, label, mfcc_1…mfcc_13, zcr, spectral_centroid,
                         spectral_bandwidth, rms_energy

Images and audio features are algorithmically derived from the text so that
the multimodal alignment is meaningful (not pure noise):
  - Image colour tint is driven by sentiment and text length.
  - Audio features simulate prosodic properties (energy, tempo) correlated
    with text statistics (punctuation, word count, etc.).

Usage:
    python scripts/generate_multimodal.py
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAW_CSV = Path("data/raw/unified_dataset.csv")
OUT_DIR = Path("data/multimodal")
IMAGE_DIR = OUT_DIR / "images"
SEED = 42


def _text_seed(text: str) -> int:
    """Deterministic integer seed from text content."""
    return int(hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest(), 16) % (2**31)


def generate_image_features(row: pd.Series) -> dict:
    """
    Generate a 32×32 synthetic image and return colour/texture features.
    Positive texts → warm (red/yellow) tones.
    Negative texts → cool (blue/grey) tones.
    Brightness and contrast are modulated by text length.
    """
    rng = np.random.default_rng(_text_seed(str(row["text"])))

    label = str(row["label"])
    text_len = len(str(row["text"]))
    brightness_base = min(0.8, 0.3 + text_len / 3000)

    if label == "pos":
        # Warm tint: high R, medium G, low B
        base_r = int(180 + rng.integers(0, 60))
        base_g = int(120 + rng.integers(0, 80))
        base_b = int(40 + rng.integers(0, 60))
    else:
        # Cool tint: low R, medium G, high B
        base_r = int(40 + rng.integers(0, 60))
        base_g = int(80 + rng.integers(0, 60))
        base_b = int(160 + rng.integers(0, 80))

    noise = rng.integers(-30, 30, size=(32, 32, 3)).astype(np.int16)
    img_arr = np.clip(
        np.array([[[base_r, base_g, base_b]]], dtype=np.int16) + noise, 0, 255
    ).astype(np.uint8)

    return {
        "img_mean_r": round(float(img_arr[:, :, 0].mean()), 3),
        "img_mean_g": round(float(img_arr[:, :, 1].mean()), 3),
        "img_mean_b": round(float(img_arr[:, :, 2].mean()), 3),
        "img_std": round(float(img_arr.std()), 3),
        "img_brightness": round(float(img_arr.mean()) / 255, 4),
        "img_contrast": round(float(img_arr.std()) / 128, 4),
        "_img_arr": img_arr,
    }


def generate_audio_features(row: pd.Series) -> dict:
    """
    Generate synthetic audio-like feature vector (MFCC + spectral features).
    Feature values are derived from text statistics to preserve some signal:
      - RMS energy ~ word count
      - Spectral centroid ~ punctuation density (exclamation / question marks)
      - MFCCs ~ character n-gram hash of text content
    """
    rng = np.random.default_rng(_text_seed(str(row["text"]) + "audio"))

    text = str(row["text"])
    label = str(row["label"])
    word_count = len(text.split())
    punct_ratio = sum(c in "!?…" for c in text) / max(len(text), 1)

    # MFCCs: 13 coefficients, label-biased means
    mfcc_means = np.array([
        (5.0 if label == "pos" else -5.0) + i * 0.5 for i in range(13)
    ])
    mfccs = mfcc_means + rng.normal(0, 2, size=13)

    rms_energy = max(0.01, 0.1 + word_count / 500 + rng.normal(0, 0.02))
    spectral_centroid = max(100, 1500 + punct_ratio * 2000 + rng.normal(0, 200))
    spectral_bandwidth = max(50, 800 + rng.normal(0, 150))
    zcr = max(0.0, 0.05 + punct_ratio * 0.3 + rng.normal(0, 0.02))

    feat = {f"mfcc_{i+1}": round(float(mfccs[i]), 4) for i in range(13)}
    feat.update({
        "zcr": round(float(zcr), 4),
        "spectral_centroid": round(float(spectral_centroid), 2),
        "spectral_bandwidth": round(float(spectral_bandwidth), 2),
        "rms_energy": round(float(rms_energy), 4),
    })
    return feat


def main():
    print(f"Loading {RAW_CSV} …")
    df = pd.read_csv(RAW_CSV)
    df = df.reset_index(drop=True)
    df["id"] = [f"item_{i:04d}" for i in range(len(df))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Text modality ──────────────────────────────────────────────────
    print("Building text modality …")
    text_df = df[["id", "text", "label", "source", "collected_at"]].copy()
    text_df["text_len"] = text_df["text"].str.len()
    text_df["word_count"] = text_df["text"].str.split().str.len()
    text_df["avg_word_len"] = (
        text_df["text_len"] / text_df["word_count"].replace(0, np.nan)
    ).round(2)
    text_df.to_csv(OUT_DIR / "text_modality.csv", index=False)
    print(f"  → {OUT_DIR}/text_modality.csv  ({len(text_df)} rows)")

    # ── Image modality ─────────────────────────────────────────────────
    print("Building image modality (32×32 synthetic images) …")
    img_rows = []
    for _, row in df.iterrows():
        feat = generate_image_features(row)
        img_arr = feat.pop("_img_arr")

        img_path = IMAGE_DIR / f"{row['id']}.png"
        Image.fromarray(img_arr).save(img_path)

        img_rows.append({"id": row["id"], "label": row["label"],
                         "img_path": str(img_path), **feat})

    image_df = pd.DataFrame(img_rows)
    image_df.to_csv(OUT_DIR / "image_modality.csv", index=False)
    print(f"  → {OUT_DIR}/image_modality.csv  ({len(image_df)} rows)")
    print(f"  → {IMAGE_DIR}/ ({len(list(IMAGE_DIR.glob('*.png')))} PNG files)")

    # ── Audio modality ─────────────────────────────────────────────────
    print("Building audio modality (synthetic MFCC features) …")
    audio_rows = []
    for _, row in df.iterrows():
        feat = generate_audio_features(row)
        audio_rows.append({"id": row["id"], "label": row["label"], **feat})

    audio_df = pd.DataFrame(audio_rows)
    audio_df.to_csv(OUT_DIR / "audio_modality.csv", index=False)
    print(f"  → {OUT_DIR}/audio_modality.csv  ({len(audio_df)} rows)")

    print("\nDone. Summary:")
    print(f"  Text modality:  {len(text_df)} rows × {len(text_df.columns)} cols")
    print(f"  Image modality: {len(image_df)} rows × {len(image_df.columns)} cols")
    print(f"  Audio modality: {len(audio_df)} rows × {len(audio_df.columns)} cols")


if __name__ == "__main__":
    main()

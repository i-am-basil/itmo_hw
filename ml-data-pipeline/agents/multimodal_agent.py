"""MultimodalAgent — loads, aligns, describes, and exports multimodal datasets.

Technical contract:
    from agents.multimodal_agent import MultimodalAgent

    agent = MultimodalAgent()

    text_data  = agent.load_modality('data/multimodal/text_modality.csv', 'text')
    image_data = agent.load_modality('data/multimodal/image_modality.csv', 'image')
    audio_data = agent.load_modality('data/multimodal/audio_modality.csv', 'audio')

    aligned = agent.align({'text': text_data, 'image': image_data, 'audio': audio_data}, key='id')

    report = agent.describe(aligned)
    # → {'n_rows': N, 'modalities': [...], 'label_dist': {...}, 'coverage': {...}, ...}

    agent.export(aligned, format='csv')   # → data/multimodal/aligned_dataset.csv
    agent.export(aligned, format='parquet')
"""

import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


# Recognised feature groups per modality type
_MODALITY_FEATURE_COLS: dict[str, list[str]] = {
    "text": ["text_len", "word_count", "avg_word_len"],
    "image": ["img_mean_r", "img_mean_g", "img_mean_b", "img_std", "img_brightness", "img_contrast"],
    "audio": [
        "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
        "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
        "mfcc_11", "mfcc_12", "mfcc_13",
        "zcr", "spectral_centroid", "spectral_bandwidth", "rms_energy",
    ],
}


class ModalityData:
    """Container for a single modality's data frame + metadata."""

    def __init__(self, df: pd.DataFrame, modality_type: str, source_path: str = ""):
        self.df = df
        self.modality_type = modality_type
        self.source_path = source_path
        # Infer feature columns that actually exist
        expected = _MODALITY_FEATURE_COLS.get(modality_type, [])
        self.feature_cols = [c for c in expected if c in df.columns]

    def __repr__(self) -> str:
        return (
            f"ModalityData(type={self.modality_type!r}, "
            f"rows={len(self.df)}, features={self.feature_cols})"
        )


class EDAReport(dict):
    """EDA report dict with a pretty-print helper."""

    def summary(self) -> str:
        lines = ["=== EDA Report ==="]
        lines.append(f"Rows:       {self.get('n_rows')}")
        lines.append(f"Modalities: {self.get('modalities')}")
        lines.append(f"Label dist: {self.get('label_dist')}")
        lines.append(f"Coverage:   {self.get('coverage')}")
        for mod, stats in self.get("modality_stats", {}).items():
            lines.append(f"\n[{mod}]")
            for k, v in stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return json.dumps(self, indent=2, default=str)


class MultimodalAgent:
    """Loads, aligns, describes, and exports multimodal datasets."""

    def __init__(self, out_dir: str = "data/multimodal"):
        self.out_dir = Path(out_dir)

    # ------------------------------------------------------------------ #
    # load_modality                                                        #
    # ------------------------------------------------------------------ #
    def load_modality(
        self,
        path: str,
        modality_type: Literal["text", "image", "audio"],
    ) -> ModalityData:
        """
        Load a modality CSV file and return a ModalityData container.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        modality_type : str
            One of 'text', 'image', 'audio'.

        Returns
        -------
        ModalityData
        """
        df = pd.read_csv(path)
        if "id" not in df.columns:
            raise ValueError(f"Modality file '{path}' must have an 'id' column.")
        mod = ModalityData(df, modality_type=modality_type, source_path=path)
        print(f"[MultimodalAgent] Loaded {modality_type} modality: {df.shape} from {path}")
        return mod

    # ------------------------------------------------------------------ #
    # align                                                                #
    # ------------------------------------------------------------------ #
    def align(
        self,
        modalities: dict[str, ModalityData],
        key: str = "id",
        how: str = "inner",
    ) -> pd.DataFrame:
        """
        Align multiple modalities by a shared key column.

        Parameters
        ----------
        modalities : dict[str, ModalityData]
            Mapping of modality name → ModalityData.
        key : str
            Column to join on (default 'id').
        how : str
            Join strategy: 'inner' (default) keeps only rows present in all
            modalities; 'outer' keeps all rows.

        Returns
        -------
        pd.DataFrame : aligned dataset with column prefixes per modality.
        """
        dfs = []
        for name, mod in modalities.items():
            df = mod.df.copy()
            # Rename feature columns to include modality prefix (avoid clashes)
            rename = {}
            skip = {key, "label", "source", "collected_at"}
            for col in df.columns:
                if col not in skip:
                    rename[col] = f"{name}__{col}"
            df = df.rename(columns=rename)
            dfs.append((name, df))

        # Start with the first modality, join the rest
        aligned = dfs[0][1]
        for name, df in dfs[1:]:
            # Determine which columns to keep (avoid duplicate label, source, etc.)
            label_cols = [c for c in df.columns if c in ("label", "source", "collected_at")]
            drop_cols = [c for c in label_cols if c in aligned.columns]
            df = df.drop(columns=drop_cols, errors="ignore")
            aligned = aligned.merge(df, on=key, how=how)

        aligned = aligned.reset_index(drop=True)
        print(
            f"[MultimodalAgent] Aligned {len(modalities)} modalities "
            f"→ {aligned.shape[0]} rows × {aligned.shape[1]} cols  (join='{how}')"
        )
        return aligned

    # ------------------------------------------------------------------ #
    # describe                                                             #
    # ------------------------------------------------------------------ #
    def describe(self, aligned_df: pd.DataFrame) -> EDAReport:
        """
        Compute an EDA report for the aligned multimodal dataset.

        Returns EDAReport dict with keys:
            n_rows, modalities, label_dist, label_dist_pct,
            coverage, modality_stats, cross_modal_corr
        """
        report = EDAReport()
        report["n_rows"] = int(len(aligned_df))

        # Detect modality prefixes from column names
        prefixes: list[str] = []
        for col in aligned_df.columns:
            if "__" in col:
                prefix = col.split("__")[0]
                if prefix not in prefixes:
                    prefixes.append(prefix)
        report["modalities"] = prefixes

        # Label distribution
        if "label" in aligned_df.columns:
            dist = aligned_df["label"].value_counts().to_dict()
            report["label_dist"] = {str(k): int(v) for k, v in dist.items()}
            total = len(aligned_df)
            report["label_dist_pct"] = {
                str(k): round(100 * v / total, 1) for k, v in dist.items()
            }

        # Coverage (non-null rate per modality)
        coverage: dict[str, float] = {}
        for prefix in prefixes:
            cols = [c for c in aligned_df.columns if c.startswith(f"{prefix}__")]
            if cols:
                coverage[prefix] = round(
                    float(aligned_df[cols].notna().all(axis=1).mean() * 100), 1
                )
        report["coverage"] = coverage

        # Per-modality feature statistics
        modality_stats: dict = {}
        for prefix in prefixes:
            cols = [c for c in aligned_df.columns if c.startswith(f"{prefix}__")]
            if not cols:
                continue
            sub = aligned_df[cols].rename(columns=lambda c: c.replace(f"{prefix}__", ""))
            numeric = sub.select_dtypes(include=[np.number])
            stats: dict = {}
            if not numeric.empty:
                stats["numeric_features"] = list(numeric.columns)
                stats["mean"] = numeric.mean().round(3).to_dict()
                stats["std"] = numeric.std().round(3).to_dict()
                stats["missing_pct"] = (
                    (numeric.isnull().sum() / len(aligned_df) * 100).round(1).to_dict()
                )
            modality_stats[prefix] = stats
        report["modality_stats"] = modality_stats

        # Cross-modal correlations with label (numeric encoding)
        if "label" in aligned_df.columns:
            label_enc = (aligned_df["label"] == "pos").astype(int)
            all_numeric_cols = [
                c for c in aligned_df.select_dtypes(include=[np.number]).columns
                if "__" in c
            ]
            corr_with_label: dict[str, float] = {}
            for col in all_numeric_cols:
                valid = aligned_df[col].dropna()
                common_idx = valid.index.intersection(label_enc.index)
                if len(common_idx) > 10:
                    r = float(np.corrcoef(valid.loc[common_idx], label_enc.loc[common_idx])[0, 1])
                    if not np.isnan(r):
                        corr_with_label[col] = round(r, 4)
            # Keep top 10 by absolute correlation
            sorted_corr = sorted(corr_with_label.items(), key=lambda x: abs(x[1]), reverse=True)
            report["top_label_correlations"] = dict(sorted_corr[:10])

        return report

    # ------------------------------------------------------------------ #
    # export                                                               #
    # ------------------------------------------------------------------ #
    def export(
        self,
        aligned_df: pd.DataFrame,
        format: Literal["csv", "parquet"] = "csv",
        out_path: Optional[str] = None,
    ) -> str:
        """
        Export the aligned dataset to disk.

        Parameters
        ----------
        aligned_df : pd.DataFrame
        format : 'csv' or 'parquet'
        out_path : optional explicit output path

        Returns
        -------
        str : path of the written file.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if out_path is None:
            fname = f"aligned_dataset.{format}"
            out_path = str(self.out_dir / fname)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            aligned_df.to_csv(out_path, index=False)
        elif format == "parquet":
            try:
                aligned_df.to_parquet(out_path, index=False)
            except ImportError:
                # Fallback if pyarrow/fastparquet not installed
                csv_path = out_path.replace(".parquet", ".csv")
                aligned_df.to_csv(csv_path, index=False)
                out_path = csv_path
                print("[MultimodalAgent] pyarrow not found — saved as CSV instead")
        else:
            raise ValueError(f"Unsupported format: {format!r}. Use 'csv' or 'parquet'.")

        print(
            f"[MultimodalAgent] Exported {len(aligned_df):,} rows "
            f"→ {out_path} ({format.upper()})"
        )
        return out_path


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MultimodalAgent CLI")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("align")
    p.add_argument("--text",  required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out",   default="data/multimodal/aligned_dataset.csv")
    p.add_argument("--how",   default="inner", choices=["inner", "outer"])

    p = sub.add_parser("describe")
    p.add_argument("--input", required=True)
    p.add_argument("--out",   default=None)

    p = sub.add_parser("export")
    p.add_argument("--input",  required=True)
    p.add_argument("--format", default="csv", choices=["csv", "parquet"])
    p.add_argument("--out",    default=None)

    args = parser.parse_args()
    agent = MultimodalAgent()

    if args.cmd == "align":
        text_mod  = agent.load_modality(args.text,  "text")
        image_mod = agent.load_modality(args.image, "image")
        audio_mod = agent.load_modality(args.audio, "audio")
        aligned = agent.align(
            {"text": text_mod, "image": image_mod, "audio": audio_mod},
            how=args.how,
        )
        agent.export(aligned, format="csv", out_path=args.out)

    elif args.cmd == "describe":
        aligned = pd.read_csv(args.input)
        report = agent.describe(aligned)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2, default=str)
        print(report.summary())

    elif args.cmd == "export":
        aligned = pd.read_csv(args.input)
        agent.export(aligned, format=args.format, out_path=args.out)

    else:
        parser.print_help()

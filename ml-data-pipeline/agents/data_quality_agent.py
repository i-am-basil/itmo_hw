"""DataQualityAgent — detects and fixes data quality problems.

Technical contract:
    from agents.data_quality_agent import DataQualityAgent

    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    # → {'missing': {...}, 'duplicates': N, 'outliers': [...], 'imbalance': {...}}

    df_clean = agent.fix(df, strategy={
        'missing': 'median',   # or 'drop', 'fill'
        'duplicates': 'drop',  # or 'keep_first'
        'outliers': 'clip_iqr' # or 'drop', 'zscore_drop', 'keep'
    })

    comparison = agent.compare(df, df_clean)
    # → dict with before/after metrics per issue type
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class QualityReport(dict):
    """Thin dict subclass so isinstance checks pass and __repr__ is readable."""

    def __repr__(self) -> str:
        return json.dumps(self, indent=2, default=str)


class ComparisonReport(dict):
    """Before/after comparison result."""

    def to_markdown(self) -> str:
        b, a = self["before"], self["after"]
        rows_removed = self["rows_removed"]
        rows_removed_pct = self["rows_removed_pct"]
        lines = [
            "| Metric | Before | After | Change |",
            "|--------|--------|-------|--------|",
            f"| Rows | {b['rows']:,} | {a['rows']:,} | -{rows_removed_pct:.1f}% |",
        ]
        metric_labels = [
            ("missing_count", "Missing values (total)"),
            ("duplicates", "Duplicate rows"),
            ("outlier_rows", "Outlier rows (text len IQR)"),
            ("imbalance_ratio", "Class imbalance ratio"),
            ("median_text_len", "Median text length"),
        ]
        for key, label in metric_labels:
            if key in b and key in a:
                bv, av = b[key], a[key]
                if isinstance(bv, float):
                    bv_s, av_s = f"{bv:.2f}", f"{av:.2f}"
                else:
                    bv_s, av_s = str(bv), str(av)
                if av < bv:
                    change = "✓ improved"
                elif av == bv:
                    change = "—"
                else:
                    change = "↑ increased"
                lines.append(f"| {label} | {bv_s} | {av_s} | {change} |")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return json.dumps(self, indent=2, default=str)


class DataQualityAgent:
    """Detects and fixes data quality issues in a pandas DataFrame."""

    # ------------------------------------------------------------------ #
    # detect_issues                                                        #
    # ------------------------------------------------------------------ #
    def detect_issues(self, df: pd.DataFrame) -> QualityReport:
        """
        Scan df for quality problems.

        Returns QualityReport (dict) with keys:
            'missing':    {col: {'count': N, 'pct': X}},
            'duplicates': N,
            'outliers':   [{'col': ..., 'count': N, 'lower': X, 'upper': Y}],
            'imbalance':  {'ratio': X, 'majority': lbl, 'minority': lbl,
                           'counts': {lbl: N, ...}}
        """
        report = QualityReport(
            missing={},
            duplicates=0,
            outliers=[],
            imbalance={},
        )

        # 1. Missing values
        for col in df.columns:
            n = int(df[col].isnull().sum())
            if n:
                report["missing"][col] = {
                    "count": n,
                    "pct": round(100 * n / len(df), 2),
                }

        # 2. Exact duplicates
        report["duplicates"] = int(df.duplicated().sum())

        # 3. Outliers — IQR on numeric columns + text length
        df_work = df.copy()
        numeric_cols = list(df_work.select_dtypes(include=[np.number]).columns)
        text_col = self._text_col(df)
        if text_col:
            df_work["_text_len"] = df_work[text_col].str.len()
            numeric_cols = ["_text_len"] + numeric_cols

        for col in numeric_cols:
            series = df_work[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((series < lower) | (series > upper)).sum())
            if n_out:
                report["outliers"].append(
                    {
                        "col": col,
                        "count": n_out,
                        "lower": round(float(lower), 2),
                        "upper": round(float(upper), 2),
                    }
                )

        # 4. Class imbalance (label column)
        label_col = self._label_col(df)
        if label_col is not None:
            counts = df[label_col].value_counts()
            if len(counts) >= 2:
                majority = str(counts.index[0])
                minority = str(counts.index[-1])
                ratio = round(float(counts.iloc[0] / counts.iloc[-1]), 3)
                report["imbalance"] = {
                    "ratio": ratio,
                    "majority": majority,
                    "minority": minority,
                    "counts": {str(k): int(v) for k, v in counts.items()},
                }

        return report

    # ------------------------------------------------------------------ #
    # fix                                                                  #
    # ------------------------------------------------------------------ #
    def fix(self, df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """
        Apply cleaning strategies to df and return a cleaned copy.

        strategy keys (all optional):
            'missing':    'drop' | 'fill' | 'median'
            'duplicates': 'drop' | 'keep_first'
            'outliers':   'clip_iqr' | 'drop' | 'zscore_drop' | 'keep'
        """
        df = df.copy()
        text_col = self._text_col(df)
        label_col = self._label_col(df)
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        key_cols = [c for c in [text_col, label_col] if c]

        # --- missing ---
        miss_strat = strategy.get("missing", "drop")
        n_before = len(df)
        if miss_strat == "drop":
            drop_subset = key_cols if key_cols else list(df.columns)
            df = df.dropna(subset=drop_subset)
        elif miss_strat in ("fill", "median"):
            if text_col:
                df[text_col] = df[text_col].fillna("[missing]")
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        print(
            f"  [fix] missing ({miss_strat}): {n_before - len(df)} rows removed",
            file=sys.stderr,
        )

        # --- duplicates ---
        dup_strat = strategy.get("duplicates", "keep_first")
        n_before = len(df)
        dedup_on = key_cols if key_cols else None
        if dup_strat in ("drop", "keep_first"):
            df = df.drop_duplicates(subset=dedup_on, keep="first")
        print(
            f"  [fix] duplicates ({dup_strat}): {n_before - len(df)} rows removed",
            file=sys.stderr,
        )

        # --- outliers ---
        out_strat = strategy.get("outliers", "clip_iqr")
        if out_strat != "keep":
            candidates = []
            if text_col:
                df["_text_len"] = df[text_col].str.len()
                candidates.append("_text_len")
            candidates += [c for c in numeric_cols if c != "_text_len"]

            n_before = len(df)
            outlier_mask = pd.Series(True, index=df.index)

            for col in candidates:
                series = df[col].dropna()
                if len(series) < 4:
                    continue
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

                if out_strat == "clip_iqr":
                    df[col] = df[col].clip(lower=lower, upper=upper)
                elif out_strat == "drop":
                    outlier_mask &= df[col].between(lower, upper) | df[col].isna()
                elif out_strat == "zscore_drop":
                    z = (df[col] - series.mean()) / (series.std() + 1e-9)
                    outlier_mask &= (z.abs() <= 3) | df[col].isna()

            if out_strat in ("drop", "zscore_drop"):
                df = df[outlier_mask]

            if "_text_len" in df.columns:
                df = df.drop(columns=["_text_len"])

            print(
                f"  [fix] outliers ({out_strat}): {n_before - len(df)} rows affected",
                file=sys.stderr,
            )

        if len(df) < 100:
            print(
                f"  WARNING: only {len(df)} rows remain after cleaning.",
                file=sys.stderr,
            )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # compare                                                              #
    # ------------------------------------------------------------------ #
    def compare(
        self, df_before: pd.DataFrame, df_after: pd.DataFrame
    ) -> ComparisonReport:
        """Return a ComparisonReport with before/after statistics."""

        def _stats(df: pd.DataFrame) -> dict:
            s: dict = {"rows": len(df)}
            text_col = self._text_col(df)
            label_col = self._label_col(df)

            # Missing
            s["missing_count"] = int(df.isnull().sum().sum())

            # Duplicates
            s["duplicates"] = int(df.duplicated().sum())

            # Text-length outliers
            if text_col:
                lengths = df[text_col].dropna().str.len()
                q1, q3 = lengths.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    s["outlier_rows"] = int(
                        ((lengths < lower) | (lengths > upper)).sum()
                    )
                else:
                    s["outlier_rows"] = 0
                s["median_text_len"] = int(lengths.median())
                s["mean_text_len"] = round(float(lengths.mean()), 1)

            # Imbalance
            if label_col is not None:
                counts = df[label_col].value_counts()
                if len(counts) >= 2:
                    s["imbalance_ratio"] = round(
                        float(counts.iloc[0] / counts.iloc[-1]), 3
                    )
                    s["n_classes"] = int(df[label_col].nunique())

            return s

        before = _stats(df_before)
        after = _stats(df_after)
        removed = before["rows"] - after["rows"]

        return ComparisonReport(
            before=before,
            after=after,
            rows_removed=removed,
            rows_removed_pct=round(100 * removed / before["rows"], 1),
        )

    # ------------------------------------------------------------------ #
    # profile                                                              #
    # ------------------------------------------------------------------ #
    def profile(self, df: pd.DataFrame) -> None:
        """Print a quick data profile to stdout."""
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
        print(f"\nDuplicates: {df.duplicated().sum()}")
        text_col = self._text_col(df)
        if text_col:
            L = df[text_col].dropna().str.len()
            print(
                f"\nText length — min={L.min()} mean={L.mean():.0f} "
                f"max={L.max()} p95={L.quantile(0.95):.0f}"
            )
        label_col = self._label_col(df)
        if label_col is not None:
            print(f"\nLabel distribution:\n{df[label_col].value_counts().to_string()}")

    def render_comparison_table(self, comparison: ComparisonReport) -> str:
        """Render the comparison as a Markdown table string."""
        if isinstance(comparison, dict) and not isinstance(
            comparison, ComparisonReport
        ):
            comparison = ComparisonReport(**comparison)
        return comparison.to_markdown()

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #
    def _text_col(self, df: pd.DataFrame) -> Optional[str]:
        for name in ("text", "Text", "content", "review", "sentence"):
            if name in df.columns:
                return name
        return None

    def _label_col(self, df: pd.DataFrame) -> Optional[str]:
        for name in ("label", "Label", "target", "class", "sentiment"):
            if name in df.columns:
                return name
        return None


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataQualityAgent CLI")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("profile")
    p.add_argument("--input", required=True)

    p = sub.add_parser("detect")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="reports/issues.json")

    p = sub.add_parser("fix")
    p.add_argument("--input", required=True)
    p.add_argument(
        "--strategy",
        required=True,
        help="Preset name (aggressive/conservative/balanced) or JSON string",
    )
    p.add_argument("--out", required=True)

    p = sub.add_parser("compare")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--out", default="reports/quality_report.md")

    args = parser.parse_args()
    agent = DataQualityAgent()

    if args.cmd == "profile":
        agent.profile(pd.read_csv(args.input))

    elif args.cmd == "detect":
        df = pd.read_csv(args.input)
        report = agent.detect_issues(df)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nduplicates : {report['duplicates']}")
        print(f"missing    : {report['missing']}")
        print(f"outliers   : {report['outliers']}")
        print(f"imbalance  : {report['imbalance']}")

    elif args.cmd == "fix":
        df = pd.read_csv(args.input)
        presets = {
            "aggressive": {
                "missing": "drop",
                "duplicates": "drop",
                "outliers": "drop",
            },
            "conservative": {
                "missing": "fill",
                "duplicates": "keep_first",
                "outliers": "keep",
            },
            "balanced": {
                "missing": "drop",
                "duplicates": "keep_first",
                "outliers": "clip_iqr",
            },
        }
        if args.strategy in presets:
            strategy = presets[args.strategy]
        else:
            strategy = json.loads(args.strategy)
        df_clean = agent.fix(df, strategy)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(args.out, index=False)
        print(f"Saved {len(df_clean):,} rows to {args.out}")

    elif args.cmd == "compare":
        df_b = pd.read_csv(args.before)
        df_a = pd.read_csv(args.after)
        comparison = agent.compare(df_b, df_a)
        table = comparison.to_markdown()
        print(table)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write("# Data Quality Report\n\n" + table)
    else:
        parser.print_help()

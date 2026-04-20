"""ActiveLearningAgent — selects the most informative examples for labeling
using uncertainty sampling with a bootstrapped logistic regression classifier.

Technical contract:
    from agents.al_agent import ActiveLearningAgent

    agent = ActiveLearningAgent(n_iterations=3, batch_size=50, random_state=42)

    # Run full AL cycle: trains on labeled seed, scores unlabeled pool
    result = agent.run_cycle(df, label_col='label', text_col='text')
    # → ALResult with .selected (DataFrame), .metrics, .uncertainty_scores

    # Score only (no new labels)
    scores = agent.score_uncertainty(df, label_col='label', text_col='text')
    # → pd.Series of per-row uncertainty values (0 = confident, 1 = max uncertain)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


@dataclass
class ALResult:
    """Result returned by run_cycle."""

    selected: pd.DataFrame
    metrics: dict
    uncertainty_scores: pd.Series
    n_iterations: int

    def summary(self) -> str:
        lines = [
            "=== ActiveLearning Result ===",
            f"  Iterations:          {self.n_iterations}",
            f"  Selected examples:   {len(self.selected)}",
        ]
        for k, v in self.metrics.items():
            lines.append(f"  {k:<24}: {v}")
        return "\n".join(lines)


class ActiveLearningAgent:
    """Uncertainty-sampling active learning agent.

    Parameters
    ----------
    n_iterations : int
        Number of AL rounds (each round trains on current labeled set,
        selects ``batch_size`` most uncertain examples and adds them).
    batch_size : int
        Number of examples to select per iteration.
    random_state : int
        Seed for reproducibility.
    min_labeled : int
        Minimum seed examples per class needed before AL starts.
    """

    def __init__(
        self,
        n_iterations: int = 3,
        batch_size: int = 50,
        random_state: int = 42,
        min_labeled: int = 20,
    ):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.min_labeled = min_labeled
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._model: Optional[LogisticRegression] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        text_col: str = "text",
        confidence_col: Optional[str] = "confidence",
    ) -> ALResult:
        """Run the full active-learning cycle.

        Uses rows where ``label_col`` is not null as the labeled seed.
        Rows without a label (or with ``label_col`` == '') are treated
        as the unlabeled pool.  If all rows are labeled, the agent still
        scores them and returns the most uncertain subset.
        """
        df = df.copy()

        # Separate labeled / unlabeled
        has_label = df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")
        labeled = df[has_label].copy()
        unlabeled = df[~has_label].copy()

        # If pool is empty, use low-confidence labeled examples as "pseudo-pool"
        if len(unlabeled) == 0:
            if confidence_col and confidence_col in df.columns:
                threshold = df[confidence_col].quantile(0.3)
                unlabeled = labeled[labeled[confidence_col] <= threshold].copy()
                labeled = labeled[labeled[confidence_col] > threshold].copy()
            else:
                # Fall back: split lowest-confidence 30% by uncertainty
                rng = np.random.default_rng(self.random_state)
                n_pool = max(self.batch_size * self.n_iterations, len(df) // 4)
                pool_idx = rng.choice(len(df), size=min(n_pool, len(df)), replace=False)
                mask = np.zeros(len(df), dtype=bool)
                mask[pool_idx] = True
                unlabeled = labeled[mask].copy()
                labeled = labeled[~mask].copy()

        # Vectorize texts
        texts_all = pd.concat([labeled[text_col], unlabeled[text_col]], ignore_index=True)
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._vectorizer.fit(texts_all.fillna("").astype(str))

        selected_indices: list = []
        iter_metrics: list[dict] = []

        current_labeled = labeled.copy()
        remaining_pool = unlabeled.copy()

        for it in range(self.n_iterations):
            if len(current_labeled) < self.min_labeled or len(remaining_pool) == 0:
                break

            # Train on current labeled set
            X_train = self._vectorizer.transform(
                current_labeled[text_col].fillna("").astype(str)
            )
            y_train = current_labeled[label_col].astype(str)

            self._model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                C=1.0,
                class_weight="balanced",
            )
            self._model.fit(X_train, y_train)

            # Evaluate on a held-out split of labeled data (if large enough)
            eval_metrics: dict = {}
            if len(current_labeled) >= 40:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train,
                    test_size=0.25,
                    random_state=self.random_state,
                    stratify=y_train if y_train.nunique() > 1 else None,
                )
                clf_eval = LogisticRegression(
                    max_iter=1000, random_state=self.random_state, class_weight="balanced"
                )
                clf_eval.fit(X_tr, y_tr)
                preds = clf_eval.predict(X_val)
                eval_metrics = {
                    "accuracy": round(accuracy_score(y_val, preds), 4),
                    "f1_macro": round(f1_score(y_val, preds, average="macro", zero_division=0), 4),
                }

            # Score the pool: uncertainty = 1 - max(proba)
            X_pool = self._vectorizer.transform(
                remaining_pool[text_col].fillna("").astype(str)
            )
            proba = self._model.predict_proba(X_pool)
            uncertainty = 1.0 - proba.max(axis=1)

            # Select top-k most uncertain
            n_select = min(self.batch_size, len(remaining_pool))
            top_idx = np.argsort(uncertainty)[-n_select:][::-1]
            selected_batch = remaining_pool.iloc[top_idx].copy()
            selected_batch["al_uncertainty"] = uncertainty[top_idx]
            selected_batch["al_iteration"] = it + 1
            selected_indices.append(selected_batch)

            iter_metrics.append(
                {
                    "iteration": it + 1,
                    "labeled_size": len(current_labeled),
                    "selected": n_select,
                    "mean_uncertainty": round(float(uncertainty[top_idx].mean()), 4),
                    **eval_metrics,
                }
            )

            # Move selected into labeled pool for next iteration
            # (treat their existing labels as ground truth)
            selected_labeled = selected_batch[selected_batch[label_col].notna()].copy()
            current_labeled = pd.concat([current_labeled, selected_labeled], ignore_index=True)
            remaining_pool = remaining_pool.drop(remaining_pool.index[top_idx])

        # Score entire df for the final uncertainty column
        X_all = self._vectorizer.transform(df[text_col].fillna("").astype(str))
        if self._model is not None:
            proba_all = self._model.predict_proba(X_all)
            unc_all = pd.Series(1.0 - proba_all.max(axis=1), index=df.index, name="al_uncertainty")
        else:
            unc_all = pd.Series(np.zeros(len(df)), index=df.index, name="al_uncertainty")

        selected_df = (
            pd.concat(selected_indices, ignore_index=True)
            if selected_indices
            else df.head(self.batch_size).copy()
        )

        return ALResult(
            selected=selected_df,
            metrics={"iterations": iter_metrics},
            uncertainty_scores=unc_all,
            n_iterations=self.n_iterations,
        )

    def score_uncertainty(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        text_col: str = "text",
    ) -> pd.Series:
        """Quick uncertainty scoring without full AL cycle."""
        result = self.run_cycle(df, label_col=label_col, text_col=text_col)
        return result.uncertainty_scores

    def save_report(self, result: ALResult, path: str = "reports/al_report.md") -> str:
        """Write AL report markdown."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Active Learning Report\n",
            f"- Iterations: {result.n_iterations}\n",
            f"- Selected total: {len(result.selected)}\n\n",
            "## Per-iteration metrics\n\n",
            "| Iteration | Labeled size | Selected | Mean uncertainty | Accuracy | F1 macro |\n",
            "|-----------|-------------|----------|-----------------|----------|----------|\n",
        ]
        for m in result.metrics.get("iterations", []):
            lines.append(
                f"| {m['iteration']} | {m['labeled_size']} | {m['selected']} "
                f"| {m['mean_uncertainty']} "
                f"| {m.get('accuracy', '-')} | {m.get('f1_macro', '-')} |\n"
            )
        Path(path).write_text("".join(lines), encoding="utf-8")
        return path

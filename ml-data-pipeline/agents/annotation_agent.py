"""AnnotationAgent — auto-labels data, generates annotation specs,
checks quality, and exports to LabelStudio.

Technical contract:
    from agents.annotation_agent import AnnotationAgent

    agent = AnnotationAgent(modality='text')
    df_labeled = agent.auto_label(df)

    spec = agent.generate_spec(df, task='sentiment_classification')
    # → annotation_spec.md written to disk, also returned as str

    metrics = agent.check_quality(df_labeled)
    # → {'kappa': 0.72, 'label_dist': {...}, 'confidence_mean': 0.85}

    agent.export_to_labelstudio(df_labeled)  # → labelstudio_import.json
    agent.export_to_labelstudio(df_labeled, flagged_only=True)  # → labelstudio_review.json
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Russian sentiment word lists (positive / negative seed words)
# Prefixes are used so that stemming variants are matched (e.g. "хорош" matches
# "хороший", "хорошо", "хорошая", etc.)
# ---------------------------------------------------------------------------
_POS_STEMS = [
    "хорош", "отличн", "прекрасн", "великолепн", "замечательн",
    "радост", "счасть", "счастлив", "победи", "победа", "успех",
    "позитивн", "добр", "любов", "красив", "лучш", "превосходн",
    "восхитительн", "улучшени", "достижени", "выигр", "доверя",
    "одобрени", "поддержк", "благодарн", "здоров", "надежд",
    "праздни", "прогресс", "развити", "спасен", "открыти",
    "достиг", "выиграл", "наград", "медал", "чемпион",
]

_NEG_STEMS = [
    "плох", "ужасн", "негативн", "проблем", "кризис",
    "смерт", "убийств", "катастроф", "авари", "трагеди",
    "конфликт", "войн", "террор", "провал", "скандал",
    "коррупц", "незакон", "арест", "задержан", "обвинен",
    "потер", "разрушен", "ухудшен", "упадк", "недовольств",
    "злост", "ненавист", "страх", "горя", "боль",
    "несчастн", "жертв", "угроз", "опасност", "вред", "ущерб",
    "погиб", "ранен", "взрыв", "пожар", "наводнен", "землетрясени",
    "осужд", "приговор", "штраф", "санкц", "запрет",
]


class AnnotationAgent:
    """Auto-labels text data and supports annotation workflow."""

    def __init__(
        self,
        modality: str = "text",
        confidence_threshold: float = 0.55,
        label_col: str = "label",
        text_col: str = "text",
        out_dir: str = "data/labeled",
    ):
        """
        Parameters
        ----------
        modality : str
            Supported: 'text'
        confidence_threshold : float
            Rows with confidence < this value are flagged for human review.
        label_col : str
            Column name for labels.
        text_col : str
            Column name for text input.
        out_dir : str
            Directory where JSON/MD output files are written.
        """
        self.modality = modality
        self.confidence_threshold = confidence_threshold
        self.label_col = label_col
        self.text_col = text_col
        self.out_dir = Path(out_dir)

    # ------------------------------------------------------------------ #
    # auto_label                                                           #
    # ------------------------------------------------------------------ #
    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zero-shot auto-labeling using TF-IDF cosine similarity to
        positive / negative sentiment seed word bags.

        Adds columns:
            auto_label      : predicted label ('pos' or 'neg')
            confidence      : float in [0, 1]
            auto_label_at   : ISO timestamp

        Only 'text' modality is currently supported.
        """
        if self.modality != "text":
            raise NotImplementedError(
                f"Modality '{self.modality}' is not yet supported. "
                "Supported: 'text'"
            )

        df = df.copy()
        texts = df[self.text_col].fillna("").tolist()

        labels, confidences = self._predict_sentiment(texts)

        df["auto_label"] = labels
        df["confidence"] = confidences
        df["auto_label_at"] = datetime.utcnow().isoformat()
        return df

    def _predict_sentiment(
        self, texts: list[str]
    ) -> tuple[list[str], list[float]]:
        """Keyword-stem counting with softmax-style confidence."""
        labels = []
        confidences = []
        for text in texts:
            text_lower = text.lower() if isinstance(text, str) else ""
            pos_hits = sum(1 for stem in _POS_STEMS if stem in text_lower)
            neg_hits = sum(1 for stem in _NEG_STEMS if stem in text_lower)

            total = pos_hits + neg_hits
            if total == 0:
                # No keyword signal — low-confidence guess based on text length
                label = "neg" if len(text_lower) > 300 else "pos"
                conf = 0.51  # intentionally below typical threshold
            else:
                raw_conf = max(pos_hits, neg_hits) / total  # in [0.5, 1.0]
                # Smooth into [0.5, 0.99] for the signal case
                conf = 0.5 + raw_conf * 0.49
                label = "pos" if pos_hits >= neg_hits else "neg"
            labels.append(label)
            confidences.append(round(float(conf), 4))

        return labels, confidences

    # ------------------------------------------------------------------ #
    # generate_spec                                                        #
    # ------------------------------------------------------------------ #
    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "sentiment_classification",
        out_path: Optional[str] = None,
    ) -> str:
        """
        Generate a Markdown annotation specification file.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset (should already have auto_label + confidence columns).
        task : str
            Task identifier used in the spec title.
        out_path : str or None
            Where to save the spec. Defaults to out_dir/annotation_spec.md.

        Returns
        -------
        str : the Markdown content
        """
        label_col = "auto_label" if "auto_label" in df.columns else self.label_col
        has_confidence = "confidence" in df.columns

        # Collect examples per class
        examples: dict[str, list[str]] = {}
        for lbl in df[label_col].dropna().unique():
            subset = df[df[label_col] == lbl].copy()
            if has_confidence:
                subset = subset.sort_values("confidence", ascending=False)
            sample = subset[self.text_col].dropna().head(3).tolist()
            examples[lbl] = sample

        # Edge cases: low-confidence samples
        edge_cases: list[str] = []
        if has_confidence:
            low_conf = df[df["confidence"] < self.confidence_threshold]
            edge_cases = low_conf[self.text_col].dropna().head(5).tolist()

        lines = [
            f"# Annotation Specification: {task.replace('_', ' ').title()}",
            "",
            f"**Дата создания:** {datetime.utcnow().strftime('%Y-%m-%d')}  ",
            f"**Модальность:** {self.modality}  ",
            f"**Датасет:** {len(df)} примеров  ",
            "",
            "---",
            "",
            "## 1. Описание задачи",
            "",
            "Задача — **бинарная классификация тональности** (sentiment analysis) "
            "коротких русскоязычных текстов.",
            "Каждый текст необходимо отнести к одному из двух классов: "
            "**pos** (позитивная тональность) или **neg** (негативная тональность).",
            "",
            "---",
            "",
            "## 2. Классы и определения",
            "",
            "### 🟢 pos — Позитивная тональность",
            "",
            "Текст выражает положительные эмоции, содержит хорошие новости, "
            "успехи, достижения, одобрение, надежду или радость.",
            "",
            "### 🔴 neg — Негативная тональность",
            "",
            "Текст выражает отрицательные эмоции, сообщает о проблемах, "
            "конфликтах, катастрофах, провалах, угрозах или горе.",
            "",
            "---",
            "",
            "## 3. Примеры разметки",
        ]

        for lbl, exs in examples.items():
            emoji = "🟢" if lbl == "pos" else "🔴"
            lines += ["", f"### {emoji} Класс `{lbl}`", ""]
            for i, ex in enumerate(exs, 1):
                short = ex[:200] + ("…" if len(ex) > 200 else "")
                lines.append(f"{i}. > {short}")

        lines += [
            "",
            "---",
            "",
            "## 4. Граничные случаи",
            "",
            "| Ситуация | Рекомендация |",
            "|----------|--------------|",
            "| Текст содержит и позитивные, и негативные факты | Выбрать преобладающий тон |",
            "| Нейтральный репортажный текст без явной оценки | Отнести к **neg** (новости чаще негативны) |",
            "| Ирония / сарказм | Разметить по буквальному смыслу, если не очевидна ирония |",
            "| Очень короткий текст (< 20 символов) | Разметить как можно точнее; при неуверенности — пропустить |",
            "| Технические артефакты (URL, цифры) | Ориентироваться на контекст, а не на артефакты |",
        ]

        if edge_cases:
            lines += [
                "",
                "---",
                "",
                "## 5. Примеры с низкой уверенностью (требуют ручной разметки)",
                "",
                f"Порог уверенности: **{self.confidence_threshold}**",
                "",
            ]
            for i, ex in enumerate(edge_cases, 1):
                short = ex[:200] + ("…" if len(ex) > 200 else "")
                lines.append(f"{i}. > {short}")

        lines += [
            "",
            "---",
            "",
            "## 6. Инструкция для разметчика",
            "",
            "1. Прочитайте текст целиком.",
            "2. Определите общую тональность: позитивная или негативная.",
            "3. Если текст явно позитивный — выберите `pos`.",
            "4. Если текст явно негативный — выберите `neg`.",
            "5. При неуверенности — ориентируйтесь на **первое впечатление**.",
            "6. Не тратьте более 10 секунд на один пример.",
            "",
            "---",
            "*Спецификация сгенерирована автоматически AnnotationAgent.*",
        ]

        content = "\n".join(lines)

        if out_path is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.out_dir / "annotation_spec.md"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(content, encoding="utf-8")
        print(f"[AnnotationAgent] Spec saved to {out_path}")
        return content

    # ------------------------------------------------------------------ #
    # check_quality                                                        #
    # ------------------------------------------------------------------ #
    def check_quality(
        self,
        df_labeled: pd.DataFrame,
        reference_col: Optional[str] = None,
    ) -> dict:
        """
        Compute quality metrics for labeled data.

        If reference_col is provided (e.g. original ground-truth labels),
        computes Cohen's κ between auto_label and reference_col.
        Otherwise computes self-agreement metrics.

        Returns
        -------
        dict with keys:
            kappa              : Cohen's κ (float or None)
            label_dist         : {label: count}
            label_dist_pct     : {label: pct}
            confidence_mean    : float
            confidence_median  : float
            low_confidence_n   : int  (rows below threshold)
            agreement_pct      : float (if reference_col given)
        """
        metrics: dict = {}

        label_col = "auto_label" if "auto_label" in df_labeled.columns else self.label_col

        # Label distribution
        dist = df_labeled[label_col].value_counts().to_dict()
        metrics["label_dist"] = {str(k): int(v) for k, v in dist.items()}
        total = len(df_labeled)
        metrics["label_dist_pct"] = {
            str(k): round(100 * v / total, 1) for k, v in dist.items()
        }

        # Confidence stats
        if "confidence" in df_labeled.columns:
            conf = df_labeled["confidence"].dropna()
            metrics["confidence_mean"] = round(float(conf.mean()), 4)
            metrics["confidence_median"] = round(float(conf.median()), 4)
            metrics["low_confidence_n"] = int(
                (conf < self.confidence_threshold).sum()
            )
        else:
            metrics["confidence_mean"] = None
            metrics["confidence_median"] = None
            metrics["low_confidence_n"] = None

        # Cohen's κ against reference
        if reference_col and reference_col in df_labeled.columns:
            y_auto = df_labeled[label_col].dropna()
            y_ref = df_labeled.loc[y_auto.index, reference_col].dropna()
            common_idx = y_auto.index.intersection(y_ref.index)
            y_auto = y_auto.loc[common_idx]
            y_ref = y_ref.loc[common_idx]
            try:
                kappa = cohen_kappa_score(y_ref, y_auto)
                metrics["kappa"] = round(float(kappa), 4)
            except Exception as e:
                metrics["kappa"] = None
                metrics["kappa_error"] = str(e)

            agreement = (y_auto.values == y_ref.values).mean()
            metrics["agreement_pct"] = round(float(agreement) * 100, 1)
        else:
            metrics["kappa"] = None
            metrics["agreement_pct"] = None

        return metrics

    # ------------------------------------------------------------------ #
    # export_to_labelstudio                                                #
    # ------------------------------------------------------------------ #
    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        flagged_only: bool = False,
        out_path: Optional[str] = None,
    ) -> list[dict]:
        """
        Export df to LabelStudio import JSON format.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain text_col. Optionally: auto_label, confidence.
        flagged_only : bool
            If True, only export rows with confidence < threshold (for review).
        out_path : str or None
            Output file path. Defaults to out_dir/labelstudio_import.json
            (or labelstudio_review.json if flagged_only=True).

        Returns
        -------
        list[dict] : LabelStudio task list
        """
        df_export = df.copy()

        # Filter to low-confidence rows if requested
        if flagged_only and "confidence" in df_export.columns:
            df_export = df_export[
                df_export["confidence"] < self.confidence_threshold
            ]

        tasks = []
        for _, row in df_export.iterrows():
            text = str(row.get(self.text_col, ""))
            task: dict = {
                "id": str(uuid.uuid4()),
                "data": {
                    "text": text,
                },
                "meta": {},
            }

            # Add source metadata if available
            for col in ("source", "collected_at"):
                if col in row.index and pd.notna(row[col]):
                    task["data"][col] = str(row[col])

            # Pre-annotation (prediction) if auto_label exists
            if "auto_label" in row.index and pd.notna(row["auto_label"]):
                score = float(row["confidence"]) if "confidence" in row.index else 1.0
                label_val = str(row["auto_label"])
                task["predictions"] = [
                    {
                        "model_version": "AnnotationAgent-v1",
                        "score": round(score, 4),
                        "result": [
                            {
                                "id": str(uuid.uuid4()),
                                "type": "choices",
                                "value": {"choices": [label_val]},
                                "from_name": "sentiment",
                                "to_name": "text",
                            }
                        ],
                    }
                ]

            # Flag low-confidence
            if "confidence" in row.index and pd.notna(row.get("confidence")):
                conf = float(row["confidence"])
                task["meta"]["confidence"] = round(conf, 4)
                task["meta"]["flagged"] = conf < self.confidence_threshold

            tasks.append(task)

        # Determine output path
        if out_path is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fname = "labelstudio_review.json" if flagged_only else "labelstudio_import.json"
            out_path = self.out_dir / fname

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(
            f"[AnnotationAgent] Exported {len(tasks)} tasks "
            f"{'(flagged only) ' if flagged_only else ''}→ {out_path}"
        )
        return tasks


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AnnotationAgent CLI")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("label")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.55)

    p = sub.add_parser("spec")
    p.add_argument("--input", required=True)
    p.add_argument("--task", default="sentiment_classification")
    p.add_argument("--out", default="data/labeled/annotation_spec.md")

    p = sub.add_parser("quality")
    p.add_argument("--input", required=True)
    p.add_argument("--reference", default=None, help="Column name of ground-truth labels")

    p = sub.add_parser("export")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--flagged-only", action="store_true")

    args = parser.parse_args()
    agent = AnnotationAgent(confidence_threshold=getattr(args, "threshold", 0.55))

    if args.cmd == "label":
        df = pd.read_csv(args.input)
        df_labeled = agent.auto_label(df)
        df_labeled.to_csv(args.out, index=False)
        print(f"Labeled {len(df_labeled)} rows → {args.out}")

    elif args.cmd == "spec":
        df = pd.read_csv(args.input)
        if "auto_label" not in df.columns:
            df = agent.auto_label(df)
        agent.generate_spec(df, task=args.task, out_path=args.out)

    elif args.cmd == "quality":
        df = pd.read_csv(args.input)
        metrics = agent.check_quality(df, reference_col=args.reference)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    elif args.cmd == "export":
        df = pd.read_csv(args.input)
        agent.export_to_labelstudio(
            df,
            flagged_only=getattr(args, "flagged_only", False),
            out_path=args.out,
        )
    else:
        parser.print_help()

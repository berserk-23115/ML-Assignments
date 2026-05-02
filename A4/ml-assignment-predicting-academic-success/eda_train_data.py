#!/usr/bin/env python3
"""Comprehensive exploratory data analysis for tabular train data.

Usage:
    python eda_train_data.py --input train_assignment.csv --output-dir eda_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extensive EDA for a CSV dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train_assignment.csv"),
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eda_output"),
        help="Directory where EDA artifacts will be written.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Target",
        help="Optional target column name for target-specific analysis.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Optional identifier column to ignore in model-focused plots.",
    )
    parser.add_argument(
        "--max-categorical-levels",
        type=int,
        default=20,
        help="Max unique values allowed for a feature to be treated as low-cardinality.",
    )
    parser.add_argument(
        "--max-grid-columns",
        type=int,
        default=12,
        help="Max number of features plotted per grid image.",
    )
    return parser.parse_args()


def chunked(values: list[str], chunk_size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), chunk_size):
        yield values[i : i + chunk_size]


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_numeric_histograms(df: pd.DataFrame, cols: list[str], output_dir: Path, chunk_size: int) -> None:
    if not cols:
        return

    for idx, sub_cols in enumerate(chunked(cols, chunk_size), start=1):
        n = len(sub_cols)
        nrows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 4.5 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, sub_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2D728F", bins=30)
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)

        for ax in axes[len(sub_cols) :]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"numeric_histograms_part_{idx}.png", dpi=180)
        plt.close(fig)


def save_numeric_boxplots(df: pd.DataFrame, cols: list[str], output_dir: Path, chunk_size: int) -> None:
    if not cols:
        return

    for idx, sub_cols in enumerate(chunked(cols, chunk_size), start=1):
        n = len(sub_cols)
        nrows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 4.5 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, sub_cols):
            sns.boxplot(x=df[col], ax=ax, color="#A6DCEF")
            ax.set_title(f"Boxplot: {col}")
            ax.set_xlabel(col)

        for ax in axes[len(sub_cols) :]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"numeric_boxplots_part_{idx}.png", dpi=180)
        plt.close(fig)


def save_low_cardinality_countplots(
    df: pd.DataFrame,
    cols: list[str],
    output_dir: Path,
    chunk_size: int,
) -> None:
    if not cols:
        return

    for idx, sub_cols in enumerate(chunked(cols, chunk_size), start=1):
        n = len(sub_cols)
        nrows = int(np.ceil(n / 2))
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(18, 5 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, sub_cols):
            ordered = df[col].value_counts(dropna=False).index
            sns.countplot(data=df, x=col, order=ordered, ax=ax, color="#F4B266")
            ax.set_title(f"Countplot: {col}")
            ax.tick_params(axis="x", rotation=45)

        for ax in axes[len(sub_cols) :]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"low_cardinality_countplots_part_{idx}.png", dpi=180)
        plt.close(fig)


def run_eda(
    df: pd.DataFrame,
    output_dir: Path | str = "eda_output",
    target_col: str | None = "Target",
    id_col: str | None = "id",
    max_categorical_levels: int = 20,
    max_grid_columns: int = 12,
    input_label: str = "in_memory_dataframe",
) -> dict:
    sns.set_theme(style="whitegrid", context="notebook")

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = df.shape
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024**2))
    duplicate_rows = int(df.duplicated().sum())
    duplicate_row_pct = round((duplicate_rows / n_rows) * 100, 4) if n_rows > 0 else 0.0
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    resolved_target_col = target_col if target_col is not None and target_col in df.columns else None
    resolved_id_col = id_col if id_col is not None and id_col in df.columns else None

    overview = {
        "input_file": str(input_label),
        "rows": n_rows,
        "columns": n_cols,
        "memory_usage_mb": round(memory_mb, 3),
        "duplicate_rows": duplicate_rows,
        "duplicate_row_pct": duplicate_row_pct,
        "columns_with_missing_values": int((missing_pct > 0).sum()),
    }
    save_json(overview, output_dir / "dataset_overview.json")

    dtype_table = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(v) for v in df.dtypes],
            "n_unique": [df[c].nunique(dropna=False) for c in df.columns],
            "missing_count": [df[c].isna().sum() for c in df.columns],
            "missing_pct": [round(df[c].isna().mean() * 100, 4) for c in df.columns],
        }
    )
    dtype_table.to_csv(tables_dir / "column_profile.csv", index=False)

    missing_table = pd.DataFrame(
        {
            "column": missing_pct.index,
            "missing_pct": missing_pct.values,
            "missing_count": [df[c].isna().sum() for c in missing_pct.index],
        }
    )
    missing_table.to_csv(tables_dir / "missingness.csv", index=False)

    plt.figure(figsize=(14, 6))
    top_missing = missing_table[missing_table["missing_pct"] > 0].head(30)
    if not top_missing.empty:
        sns.barplot(data=top_missing, x="column", y="missing_pct", color="#8CC084")
        plt.xticks(rotation=60, ha="right")
        plt.title("Top Missingness by Column")
        plt.tight_layout()
        plt.savefig(plots_dir / "missingness_top_columns.png", dpi=180)
    plt.close()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    excluded_cols = {c for c in [resolved_target_col, resolved_id_col] if c is not None}
    model_numeric_cols = [c for c in numeric_cols if c not in excluded_cols]

    if numeric_cols:
        numeric_summary = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        numeric_summary["missing_pct"] = df[numeric_cols].isna().mean() * 100
        numeric_summary["zero_pct"] = (df[numeric_cols] == 0).mean() * 100
        numeric_summary["skew"] = df[numeric_cols].skew(numeric_only=True)
        numeric_summary["kurtosis"] = df[numeric_cols].kurtosis(numeric_only=True)
        numeric_summary = numeric_summary.reset_index().rename(columns={"index": "column"})
        numeric_summary.to_csv(tables_dir / "numeric_summary.csv", index=False)

        corr = df[numeric_cols].corr(numeric_only=True)
        corr.to_csv(tables_dir / "correlation_matrix.csv")

        plt.figure(figsize=(min(22, 0.6 * len(numeric_cols) + 6), min(16, 0.6 * len(numeric_cols) + 4)))
        sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_heatmap.png", dpi=180)
        plt.close()

        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        outlier_counts = ((df[numeric_cols] < (q1 - 1.5 * iqr)) | (df[numeric_cols] > (q3 + 1.5 * iqr))).sum()
        outlier_table = pd.DataFrame(
            {
                "column": outlier_counts.index,
                "outlier_count_iqr": outlier_counts.values,
                "outlier_pct_iqr": (outlier_counts.values / len(df)) * 100,
            }
        ).sort_values("outlier_pct_iqr", ascending=False)
        outlier_table.to_csv(tables_dir / "outlier_summary_iqr.csv", index=False)

        save_numeric_histograms(df, model_numeric_cols, plots_dir, chunk_size=max(3, max_grid_columns))
        save_numeric_boxplots(df, model_numeric_cols, plots_dir, chunk_size=max(3, max_grid_columns))

    # Include string/object columns and low-cardinality numeric columns in categorical exploration.
    object_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    low_card_numeric = [
        c for c in numeric_cols if df[c].nunique(dropna=False) <= max_categorical_levels and c not in excluded_cols
    ]
    categorical_like_cols = sorted(set(object_cols + low_card_numeric) - excluded_cols)

    if categorical_like_cols:
        categorical_rows = []
        for col in categorical_like_cols:
            value_counts = df[col].value_counts(dropna=False)
            top_values = value_counts.head(10).to_dict()
            categorical_rows.append(
                {
                    "column": col,
                    "n_unique": int(df[col].nunique(dropna=False)),
                    "missing_pct": round(df[col].isna().mean() * 100, 4),
                    "mode": value_counts.index[0] if not value_counts.empty else np.nan,
                    "mode_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    "top_10_values": json.dumps(top_values),
                }
            )

        categorical_summary = pd.DataFrame(categorical_rows).sort_values("n_unique")
        categorical_summary.to_csv(tables_dir / "categorical_like_summary.csv", index=False)
        save_low_cardinality_countplots(
            df,
            categorical_like_cols,
            plots_dir,
            chunk_size=max(2, max_grid_columns // 2),
        )

    if resolved_target_col is not None:
        target_counts = df[resolved_target_col].value_counts(dropna=False)
        target_table = pd.DataFrame(
            {
                "target_class": target_counts.index,
                "count": target_counts.values,
                "pct": (target_counts.values / len(df)) * 100,
            }
        )
        target_table.to_csv(tables_dir / "target_distribution.csv", index=False)

        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=resolved_target_col, order=target_counts.index, color="#7DB7D8")
        plt.title(f"Target Distribution: {resolved_target_col}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "target_distribution.png", dpi=180)
        plt.close()

        if model_numeric_cols and df[resolved_target_col].nunique(dropna=False) <= 15:
            group_means = df.groupby(resolved_target_col)[model_numeric_cols].mean(numeric_only=True)
            group_means.to_csv(tables_dir / "target_group_numeric_means.csv")

            plt.figure(figsize=(min(20, len(model_numeric_cols) * 0.45 + 4), 6))
            sns.heatmap(group_means, cmap="YlGnBu", linewidths=0.2)
            plt.title("Mean of Numeric Features by Target Class")
            plt.tight_layout()
            plt.savefig(plots_dir / "target_group_numeric_means_heatmap.png", dpi=180)
            plt.close()

    report_lines = [
        "# EDA Report",
        "",
        f"- Input file: {input_label}",
        f"- Rows: {n_rows}",
        f"- Columns: {n_cols}",
        f"- Duplicate rows: {duplicate_rows} ({duplicate_row_pct:.4f}%)",
        f"- Columns with missing values: {(missing_pct > 0).sum()}",
        f"- Numeric columns: {len(numeric_cols)}",
        f"- Categorical-like columns: {len(categorical_like_cols)}",
    ]
    if resolved_target_col is not None:
        report_lines.append(f"- Target column detected: {resolved_target_col}")
    else:
        report_lines.append("- Target column not found; skipped target analysis")

    (output_dir / "EDA_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    overview["target_column"] = resolved_target_col
    overview["output_dir"] = str(output_dir)
    overview["tables_dir"] = str(tables_dir)
    overview["plots_dir"] = str(plots_dir)
    return overview


def main() -> None:
    args = parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    summary = run_eda(
        df,
        output_dir=args.output_dir,
        target_col=args.target_col,
        id_col=args.id_col,
        max_categorical_levels=args.max_categorical_levels,
        max_grid_columns=args.max_grid_columns,
        input_label=str(input_path),
    )

    print("EDA complete.")
    print(f"Input: {input_path}")
    print(f"Artifacts written to: {summary['output_dir']}")
    print(f"Tables: {summary['tables_dir']}")
    print(f"Plots: {summary['plots_dir']}")


if __name__ == "__main__":
    main()

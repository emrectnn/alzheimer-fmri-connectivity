"""Aggregate the classification sweep CSVs into ranked leaderboard tables."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kod"))
config = import_module("00a_config")


METRICS_DIR = Path(config.METRICS_DIR)
SWEEP_DIR = METRICS_DIR / "sweep"


def _load_all_sweep_csvs() -> pd.DataFrame:
    """Concatenate every sweep CSV into one DataFrame."""
    if not SWEEP_DIR.exists():
        return pd.DataFrame()
    frames = []
    for p in SWEEP_DIR.glob("*.csv"):
        try:
            d = pd.read_csv(p)
            d['__source__'] = p.name
            frames.append(d)
        except Exception as e:
            print(f"[WARN] {p.name} okunamadi: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _safe_str_task(t) -> str:
    """Normalize a task label to text ('A-B' for a pair tuple)."""
    if isinstance(t, tuple):
        return f"{t[0]}-{t[1]}"
    return str(t)


def build_headline(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Best-AUC row per (task, mode).

    Args:
        sweep_df: Concatenated sweep results.

    Returns:
        DataFrame with the winning row per task/mode.
    """
    if sweep_df.empty:
        return pd.DataFrame()
    cols_req = ['Task', 'Mode', 'Features', 'Model', 'AUC']
    for c in cols_req:
        if c not in sweep_df.columns:
            return pd.DataFrame()
    d = sweep_df.copy()
    d['Task'] = d['Task'].apply(_safe_str_task)
    idx = d.groupby(['Task', 'Mode'])['AUC'].idxmax()
    headline = d.loc[idx].reset_index(drop=True)
    keep = ['Task', 'Mode', 'Features', 'Model', 'AUC',
            'AUC_CI_low', 'AUC_CI_high', 'Acc', 'F1',
            'Residualized', 'ComBat', 'Optuna']
    keep = [c for c in keep if c in headline.columns]
    return headline[keep].sort_values(['Task', 'Mode']).reset_index(drop=True)


def build_alff_reho_ablation(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """ALFF/ReHo ablation: base graph vs augmented feature sets.

    Args:
        sweep_df: Concatenated sweep results.

    Returns:
        Pivoted DataFrame of AUC per feature set.
    """
    if sweep_df.empty or 'Features' not in sweep_df.columns:
        return pd.DataFrame()
    targets = ['Graf_Tam', 'Graf_Tam+ALFF', 'Graf_Tam+ReHo', 'Graf_Tam+ALFF+ReHo',
               'ALFF', 'ReHo', 'ALFF+ReHo']
    d = sweep_df[sweep_df['Features'].isin(targets)].copy()
    if d.empty:
        return pd.DataFrame()
    d['Task'] = d['Task'].apply(_safe_str_task)
    idx = d.groupby(['Task', 'Mode', 'Features'])['AUC'].idxmax()
    best = d.loc[idx]
    pivot = best.pivot_table(index=['Task', 'Mode'], columns='Features',
                             values='AUC', aggfunc='first')
    return pivot.reset_index()


def build_nbs_ablation(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Tangent_FC vs NBS_Edges comparison.

    Args:
        sweep_df: Concatenated sweep results.

    Returns:
        Pivoted DataFrame with a Delta column.
    """
    if sweep_df.empty:
        return pd.DataFrame()
    d = sweep_df[sweep_df['Features'].isin(['Tangent_FC', 'NBS_Edges'])].copy()
    if d.empty:
        return pd.DataFrame()
    d['Task'] = d['Task'].apply(_safe_str_task)
    idx = d.groupby(['Task', 'Mode', 'Features'])['AUC'].idxmax()
    best = d.loc[idx]
    pivot = best.pivot_table(index=['Task', 'Mode'], columns='Features',
                             values='AUC', aggfunc='first')
    if 'Tangent_FC' in pivot.columns and 'NBS_Edges' in pivot.columns:
        pivot['Delta_NBS_vs_Tangent'] = pivot['NBS_Edges'] - pivot['Tangent_FC']
    return pivot.reset_index()


def build_twostage_comparison(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Flat vs two-stage 3-class AUC comparison.

    Args:
        sweep_df: Concatenated sweep results.

    Returns:
        Pivoted DataFrame with a Delta column.
    """
    if sweep_df.empty:
        return pd.DataFrame()
    d = sweep_df[sweep_df['Task'].isin(['3class', '3class_twostage'])].copy()
    if d.empty:
        return pd.DataFrame()
    idx = d.groupby(['Task', 'Mode'])['AUC'].idxmax()
    best = d.loc[idx]
    pivot = best.pivot_table(index='Mode', columns='Task', values='AUC',
                             aggfunc='first')
    if '3class' in pivot.columns and '3class_twostage' in pivot.columns:
        pivot['Delta_Twostage'] = pivot['3class_twostage'] - pivot['3class']
    return pivot.reset_index()


def load_optional(path: Path) -> pd.DataFrame | None:
    """Read a CSV if present, else return None."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def df_to_md(df: pd.DataFrame, float_fmt: str = '.3f') -> str:
    """Render a DataFrame as Markdown ('(no data)' if empty)."""
    if df is None or df.empty:
        return "_(veri yok)_\n"
    try:
        return df.to_markdown(index=False, floatfmt=float_fmt) + "\n"
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```\n"


def main() -> int:
    """Build every table and write leaderboard_final.md plus the headline CSV."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sweep = _load_all_sweep_csvs()
    if sweep.empty:
        print(f"[ERR] sweep CSV bulunamadi: {SWEEP_DIR}")
        return 1
    print(f"[INFO] {len(sweep)} satir {SWEEP_DIR} altindan yuklendi")

    headline = build_headline(sweep)
    alff_reho = build_alff_reho_ablation(sweep)
    nbs = build_nbs_ablation(sweep)
    twostage = build_twostage_comparison(sweep)
    per_class = load_optional(METRICS_DIR / 'per_class_metrics.csv')
    perm = load_optional(METRICS_DIR / 'permutation_pvalues.csv')

    s3_lb = load_optional(METRICS_DIR / 'leaderboard_optuna.csv')
    if s3_lb is None:
        s3_lb = load_optional(METRICS_DIR / 'leaderboard_fixed.csv')
    delta_df = pd.DataFrame()
    if s3_lb is not None and not headline.empty:
        s3_lb = s3_lb.copy()
        s3_lb['Task'] = s3_lb['Task'].apply(_safe_str_task)
        key = ['Task', 'Mode']
        s3_best = (s3_lb.sort_values('AUC', ascending=False)
                   .groupby(key, as_index=False).head(1)
                   [['Task', 'Mode', 'AUC']].rename(columns={'AUC': 'AUC_baseline'}))
        delta_df = headline[['Task', 'Mode', 'AUC']].rename(
            columns={'AUC': 'AUC_final'}).merge(s3_best, on=key, how='left')
        delta_df['Delta'] = delta_df['AUC_final'] - delta_df['AUC_baseline']

    md_lines = []
    md_lines.append("# Leaderboard\n")
    md_lines.append("## 1. Headline (her Task × Mode icin en iyi AUC)\n")
    md_lines.append(df_to_md(headline))
    if not delta_df.empty:
        md_lines.append("## 2. Baseline vs Final Delta\n")
        md_lines.append(df_to_md(delta_df))
    md_lines.append("## 3. ALFF/ReHo ablation (Graf_Tam + ek)\n")
    md_lines.append(df_to_md(alff_reho))
    md_lines.append("## 4. NBS_Edges vs Tangent_FC\n")
    md_lines.append(df_to_md(nbs))
    md_lines.append("## 5. Two-stage (hiyerarsik) vs flat 3-class\n")
    md_lines.append(df_to_md(twostage))
    md_lines.append("## 6. Per-class metrikler\n")
    md_lines.append(df_to_md(per_class))
    md_lines.append("## 7. Permutation p-degerleri\n")
    md_lines.append(df_to_md(perm))
    md_lines.append(
        "\n---\n**Not:** `imaging_plus_clinical` satirlari LEAKAGE referansidir "
        "(MMSE/CDR tani tanimini icerir). Ana rapor `imaging_only` ve "
        "`imaging_plus_demographics` satirlarindan olusturulmalidir.\n")

    out_md = METRICS_DIR / 'leaderboard_final.md'
    out_md.write_text("\n".join(md_lines), encoding='utf-8')
    print(f"[SAVE] {out_md}")

    out_csv = METRICS_DIR / 'leaderboard_headline.csv'
    if not headline.empty:
        headline.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

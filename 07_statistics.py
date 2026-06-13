"""Group statistical comparisons (t-test, ANOVA, effect sizes) on graph metrics."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def group_comparison(metrics_df, metric_col, groups=None):
    """Compare a metric across groups with an effect size.

    Args:
        metrics_df: Per-subject metrics with a group column.
        metric_col: Name of the metric column to test.
        groups: Optional subset of group labels to compare.

    Returns:
        Dict with the test statistic, p-value and effect size.
    """
    if groups is None:
        groups = ["HC", "MCI", "AD"]

    data = {g: metrics_df[metrics_df["group"] == g][metric_col].dropna()
            for g in groups if g in metrics_df["group"].values}

    data = {g: v for g, v in data.items() if len(v) > 0}
    if len(data) < 2:
        return pd.DataFrame()

    kw_stat, kw_p = stats.kruskal(*data.values())

    pairs = [("HC", "MCI"), ("HC", "AD"), ("MCI", "AD")]
    results = []

    for g1, g2 in pairs:
        if g1 not in data or g2 not in data:
            continue
        if len(data[g1]) < 2 or len(data[g2]) < 2:
            continue

        u, p = stats.mannwhitneyu(data[g1], data[g2], alternative="two-sided")

        pooled_std = np.sqrt(
            (data[g1].std()**2 + data[g2].std()**2) / 2
        )
        d = (data[g1].mean() - data[g2].mean()) / pooled_std if pooled_std > 0 else 0

        if abs(d) >= 0.8:
            effect = "large"
        elif abs(d) >= 0.5:
            effect = "medium"
        elif abs(d) >= 0.2:
            effect = "small"
        else:
            effect = "negligible"

        results.append({
            "metric": metric_col,
            "pair": f"{g1} vs {g2}",
            "g1_mean": data[g1].mean(),
            "g1_std": data[g1].std(),
            "g2_mean": data[g2].mean(),
            "g2_std": data[g2].std(),
            "U": u,
            "p": p,
            "cohens_d": d,
            "effect_size": effect,
            "kw_H": kw_stat,
            "kw_p": kw_p,
        })

    return pd.DataFrame(results)


def report_all_metrics(metrics_df, metric_cols=None, correction="bonferroni"):
    """Run group comparisons for all metrics with correction.

    Args:
        metrics_df: Per-subject metrics table.
        metric_cols: Optional metric columns; defaults to all numeric.
        correction: Multiple-comparison correction method.

    Returns:
        DataFrame of per-metric, per-pair statistics.
    """
    if metric_cols is None:
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        exclude = ["label", "n_nodes", "n_edges", "n_nodes_lc", "n_communities"]
        metric_cols = [c for c in numeric_cols if c not in exclude]

    all_results = []
    for col in metric_cols:
        res = group_comparison(metrics_df, col)
        if not res.empty:
            all_results.append(res)

    if not all_results:
        print("Karsilastirma yapilamadi.")
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)

    if len(df) > 1:
        reject, p_adj, _, _ = multipletests(df["p"], method=correction)
        df["p_adjusted"] = p_adj
        df["significant"] = reject
    else:
        df["p_adjusted"] = df["p"]
        df["significant"] = df["p"] < 0.05

    df = df.sort_values("p_adjusted")

    return df


def print_summary(stats_df):
    """Print a short summary of the statistical results.

    Args:
        stats_df: Output of report_all_metrics().

    Returns:
        None; prints to stdout.
    """
    if stats_df.empty:
        print("Sonuc yok.")
        return

    print("ISTATISTIKSEL KARSILASTIRMA OZETI")

    sig = stats_df[stats_df["significant"]]
    print(f"\nAnlamli sonuclar: {len(sig)} / {len(stats_df)}")

    if not sig.empty:
        for _, row in sig.iterrows():
            star = "***" if row["p_adjusted"] < 0.001 else (
                "**" if row["p_adjusted"] < 0.01 else "*")
            print(f"\n  {row['metric']} ({row['pair']}) {star}")
            print(f"    p_adj = {row['p_adjusted']:.4f}, "
                  f"Cohen's d = {row['cohens_d']:.3f} ({row['effect_size']})")
            print(f"    {row['pair'].split(' vs ')[0]}: "
                  f"{row['g1_mean']:.4f} ± {row['g1_std']:.4f}")
            print(f"    {row['pair'].split(' vs ')[1]}: "
                  f"{row['g2_mean']:.4f} ± {row['g2_std']:.4f}")

    print("\n--- Tablo 5.1 Kriter Kontrolu ---")
    hc_ad = stats_df[stats_df["pair"] == "HC vs AD"]
    if not hc_ad.empty:
        min_p = hc_ad["p_adjusted"].min()
        print(f"  HC vs AD minimum p: {min_p:.4f}")
        if min_p < 0.05:
            print("  OK Kriter cible atteint (p < 0.05, Bonferroni)")
        elif min_p < 0.10:
            print("  ~ Kriter minimum atteint (p < 0.10)")
        else:
            print("  FAIL Kriter non atteint")

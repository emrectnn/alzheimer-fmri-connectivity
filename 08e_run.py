"""Sweep orchestration, leaderboard writing and plotting.

Top-level entry point (run_enhanced_classification) that runs every experiment,
assembles the leaderboards, and writes the result CSVs/figures.
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

try:
    import neuroHarmonize  # noqa: F401
    HAS_NH = True
except ImportError:
    HAS_NH = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")

# This module is the public entry point of the classification stage. It pulls
# together the API spread across 08a-08d so that external callers (main.py, the
# qc utilities and the tests) can import a single module. Names that monkeypatch
# a module attribute (e.g. harmonizationLearn) must instead target the module
# that defines them (08b_transformers).
_feat = import_module("08a_features")
_tf = import_module("08b_transformers")
_models = import_module("08c_models")
_exp = import_module("08d_experiments")

# Feature sets (08a)
load_all_features = _feat.load_all_features
get_feature_sets = _feat.get_feature_sets
get_feature_sets_by_mode = _feat.get_feature_sets_by_mode
_prepare_feature_matrix = _feat._prepare_feature_matrix

# Transformers and pipelines (08b)
ConfoundRegressor = _tf.ConfoundRegressor
ComBatTransformer = _tf.ComBatTransformer
make_pipeline = _tf.make_pipeline
make_imb_pipeline = _tf.make_imb_pipeline
make_tangent_pipeline = _tf.make_tangent_pipeline
load_timeseries_lookup = _tf.load_timeseries_lookup
HAS_IMBLEARN = _tf.HAS_IMBLEARN

# Models and cross-validation (08c)
cv_score = _models.cv_score
_build_models = _models._build_models
_build_models_optuna = _models._build_models_optuna
HAS_XGB = _models.HAS_XGB
HAS_LGBM = _models.HAS_LGBM
HAS_MORD = _models.HAS_MORD

# Experiments (08d)
experiment_3class_by_mode = _exp.experiment_3class_by_mode
experiment_binary_by_mode = _exp.experiment_binary_by_mode
experiment_twostage_by_mode = _exp.experiment_twostage_by_mode


def plot_results(res_3class, res_binary, output_dir):
    """Plot the 3-class and binary result summaries to PNG.

    Args:
        res_3class: 3-class result DataFrame.
        res_binary: Binary result DataFrame.
        output_dir: Directory for the figure.

    Returns:
        None; writes a PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top15 = res_3class.head(15)
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top15)))
    bars = ax.barh(
        range(len(top15)),
        top15['AUC'].values,
        color=colors, edgecolor='gray', linewidth=0.5
    )
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(
        [f"{r['Features'][:12]}\n{r['Model']}" for _, r in top15.iterrows()],
        fontsize=8)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Sanslilik (0.5)')
    ax.axvline(x=0.33, color='orange', linestyle=':', alpha=0.7, label='Rastgele')
    ax.set_xlabel('AUC (weighted OvR)')
    ax.set_title('3-Sinif HC/MCI/AD - AUC Karsilastirmasi\n(Ust 15 kombinasyon)')
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 0.85)

    top15b = res_binary.head(15)
    ax2 = axes[1]
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top15b)))
    ax2.barh(
        range(len(top15b)),
        top15b['AUC'].values,
        color=colors2, edgecolor='gray', linewidth=0.5
    )
    ax2.set_yticks(range(len(top15b)))
    ax2.set_yticklabels(
        [f"{r['Features'][:12]}\n{r['Model']}" for _, r in top15b.iterrows()],
        fontsize=8)
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Sanslilik (0.5)')
    ax2.set_xlabel('AUC (binary HC vs AD)')
    ax2.set_title('Binary HC vs AD - AUC Karsilastirmasi\n(Ust 15 kombinasyon)')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.3, 0.95)

    plt.tight_layout()
    path = os.path.join(output_dir, 'enhanced_classification_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGorsel kaydedildi: {path}")


def plot_confusion_matrix(df, output_dir):
    """Render and save a confusion-matrix figure.

    Args:
        df: Feature DataFrame.
        output_dir: Directory for the figure.

    Returns:
        None; writes a PNG.
    """
    img_sets = get_feature_sets_by_mode(df, 'imaging_only')
    cols = img_sets.get('Graf_AUC+Null', img_sets['Graf_AUC'])
    X = df[cols].values
    y = df['label'].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_true = [], []
    for train_idx, test_idx in cv.split(X, y):
        pipe = make_imb_pipeline(
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        )
        pipe.fit(X[train_idx], y[train_idx])
        all_preds.extend(pipe.predict(X[test_idx]))
        all_true.extend(y[test_idx])

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['HC', 'MCI', 'AD'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix - GBM (imaging_only, CV-safe)\n(Graf_AUC+Null, 5-fold CV)')
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix_best.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix kaydedildi: {path}")


def _slug(task, mode, residualize, variant=None):
    """Build a short filename slug for a (task, mode, residualize) slice."""
    t = 'task-' + (task if isinstance(task, str) else f"{task[0]}-{task[1]}")
    r = 'resid' if residualize else 'raw'
    base = f"{t}__{mode}__{r}"
    return f"{base}__{variant}" if variant else base


def build_leaderboard(results_dict, out_dir, variant=None):
    """Aggregate sweep results into a ranked leaderboard table.

    Args:
        results_dict: Mapping of (task, mode, residualize) to result frames.
        out_dir: Output directory for the CSV/Markdown.
        variant: Optional tag (e.g. 'fixed' or 'optuna').

    Returns:
        The leaderboard DataFrame.
    """
    rows = []
    for (task, mode, resid), df_res in results_dict.items():
        if df_res is None or df_res.empty:
            continue
        df_nn = df_res.dropna(subset=['AUC'])
        if df_nn.empty:
            continue
        top = df_nn.sort_values('AUC', ascending=False).iloc[0]
        rows.append({
            'Task': task if isinstance(task, str) else f"{task[0]}-{task[1]}",
            'Mode': mode,
            'Residualized': bool(resid),
            'ComBat': bool(top.get('ComBat', False)),
            'Optuna': bool(top.get('Optuna', False)),
            'Best_Features': top.get('Features', ''),
            'Best_Model': top.get('Model', ''),
            'n_feat': int(top.get('n_feat', 0)),
            'Acc': float(top.get('Acc', np.nan)),
            'AUC': float(top.get('AUC', np.nan)),
            'AUC_CI_low': float(top.get('AUC_CI_low', np.nan)),
            'AUC_CI_high': float(top.get('AUC_CI_high', np.nan)),
            'F1': float(top.get('F1', np.nan)),
            'Train_Acc': float(top.get('Train_Acc', np.nan)),
            'Leakage_Flag': (mode == 'imaging_plus_clinical'),
        })
    lb = pd.DataFrame(rows)
    if lb.empty:
        print("[WARN] Leaderboard bos — hic sonuc toplanmadi.")
        return lb

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{variant}" if variant else ""
    csv_path = os.path.join(out_dir, f"leaderboard{suffix}.csv")
    lb.to_csv(csv_path, index=False)
    print(f"\nLeaderboard CSV: {csv_path}")

    title = "Leaderboard (tuned)" if variant else "Leaderboard (fixed)"
    subtitle = {
        'fixed': "RepeatedStratifiedKFold + ComBat (sabit hiperparametreler)",
        'optuna': "RepeatedStratifiedKFold + ComBat + Optuna Bayesian HPO",
    }.get(variant, "")
    md_lines = [
        f"# {title}",
        "",
        subtitle,
        "",
        "Her (Task, Mode, Residualized) icin AUC en yuksek model.",
        "",
        "**UYARI:** `imaging_plus_clinical` modunda MMSE/CDR feature'lari target",
        "leakage icerir (tani tanimlari). Sadece upper-bound referans; tez ana",
        "sonuclari **imaging_only** satirlarindandir.",
        "",
    ]
    for task_lbl, g in lb.groupby('Task'):
        md_lines.append(f"## {task_lbl}")
        md_lines.append("")
        md_lines.append(
            "| Mode | Residualized | ComBat | Optuna | Features | Model | "
            "n_feat | Acc | AUC [95% CI] | F1 | Train_Acc | Leakage |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        g_sorted = g.sort_values(['Leakage_Flag', 'Mode', 'Residualized'])
        for _, r in g_sorted.iterrows():
            flag = "**LEAKAGE**" if r['Leakage_Flag'] else ""
            ci_low = r.get('AUC_CI_low', np.nan)
            ci_high = r.get('AUC_CI_high', np.nan)
            if pd.notna(ci_low) and pd.notna(ci_high):
                auc_str = f"**{r['AUC']:.3f}** [{ci_low:.3f}-{ci_high:.3f}]"
            else:
                auc_str = f"**{r['AUC']:.3f}**"
            md_lines.append(
                f"| {r['Mode']} | {r['Residualized']} |"
                f" {'Y' if r.get('ComBat') else '-'} |"
                f" {'Y' if r.get('Optuna') else '-'} |"
                f" {r['Best_Features']} | {r['Best_Model']} |"
                f" {r['n_feat']} | {r['Acc']:.3f} | {auc_str} |"
                f" {r['F1']:.3f} | {r['Train_Acc']:.3f} | {flag} |"
            )
        md_lines.append("")
    md_path = os.path.join(out_dir, f"leaderboard{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Leaderboard MD : {md_path}")

    return lb


def _run_sweep_pass(df, ts_lookup, tasks, modes, sweep_dir,
                    use_combat, use_optuna, n_repeats, tag):
    """Run one full sweep pass (all tasks x modes) and collect results.

    Args:
        df: Merged feature DataFrame.
        ts_lookup: id-to-array map for tangent features.
        tasks: Tasks to run.
        modes: Feature modes to run.
        sweep_dir: Directory for per-slice CSVs.
        use_combat: Apply ComBat when True.
        use_optuna: Tune with Optuna when True.
        n_repeats: CV repeats.
        tag: Variant tag for output names.

    Returns:
        Dict of result frames keyed by (task, mode, residualize).
    """
    print(f"  Pass: {tag} | ComBat={use_combat} | Optuna={use_optuna}"
          f" | n_repeats={n_repeats}")
    results = {}
    for task in tasks:
        for mode in modes:
            resid_opts = [False] if mode == 'imaging_plus_clinical' else [False, True]
            for resid in resid_opts:
                if task == '3class':
                    r = experiment_3class_by_mode(
                        df, mode, residualize=resid,
                        time_series_lookup=ts_lookup,
                        use_combat=use_combat, use_optuna=use_optuna,
                        n_repeats=n_repeats)
                elif task == '3class_twostage':
                    if mode == 'imaging_plus_clinical':
                        continue
                    try:
                        r = experiment_twostage_by_mode(
                            df, mode, residualize=resid,
                            use_combat=use_combat, n_repeats=n_repeats)
                    except Exception as e:
                        print(f"[WARN] twostage {mode} resid={resid} basarisiz: {e}")
                        r = pd.DataFrame()
                else:
                    r = experiment_binary_by_mode(
                        df, mode, pair=task, residualize=resid,
                        time_series_lookup=ts_lookup,
                        use_combat=use_combat, use_optuna=use_optuna,
                        n_repeats=n_repeats)
                results[(task, mode, resid)] = r
                slug = _slug(task, mode, resid, variant=tag)
                if r is not None and not r.empty:
                    r.to_csv(os.path.join(sweep_dir, f"{slug}.csv"),
                             index=False)
    return results


def _write_comparison(lb_fixed, lb_optuna, out_dir, baseline_ref=None):
    """Write the fixed-vs-tuned leaderboard comparison table.

    Args:
        lb_fixed: Leaderboard from the fixed-hyperparameter pass.
        lb_optuna: Leaderboard from the Optuna pass.
        out_dir: Output directory.
        baseline_ref: Optional earlier leaderboard for a delta column.

    Returns:
        None; writes a Markdown table.
    """
    md_path = os.path.join(out_dir, "leaderboard_comparison.md")
    merged = lb_fixed.rename(columns={
        'AUC': 'AUC_S3_fixed',
        'AUC_CI_low': 'CI_low_S3_fixed',
        'AUC_CI_high': 'CI_high_S3_fixed',
    })[['Task', 'Mode', 'Residualized', 'AUC_S3_fixed',
        'CI_low_S3_fixed', 'CI_high_S3_fixed']].copy()
    if not lb_optuna.empty:
        opt = lb_optuna.rename(columns={
            'AUC': 'AUC_S3_optuna',
            'AUC_CI_low': 'CI_low_S3_optuna',
            'AUC_CI_high': 'CI_high_S3_optuna',
        })[['Task', 'Mode', 'Residualized', 'AUC_S3_optuna',
            'CI_low_S3_optuna', 'CI_high_S3_optuna']]
        merged = merged.merge(opt, on=['Task', 'Mode', 'Residualized'],
                              how='outer')
    if baseline_ref is not None and not baseline_ref.empty:
        s2 = baseline_ref.rename(columns={'AUC': 'AUC_S2'})
        if 'Residualized' not in s2.columns:
            s2['Residualized'] = False
        merged = merged.merge(
            s2[['Task', 'Mode', 'Residualized', 'AUC_S2']],
            on=['Task', 'Mode', 'Residualized'], how='left')
    if 'AUC_S2' in merged.columns:
        merged['Delta_S2_to_S3fixed'] = merged['AUC_S3_fixed'] - merged['AUC_S2']
    if 'AUC_S3_optuna' in merged.columns:
        merged['Delta_S3fixed_to_S3optuna'] = merged['AUC_S3_optuna'] - merged['AUC_S3_fixed']
    merged.to_csv(md_path.replace('.md', '.csv'), index=False)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# AUC Comparison\n\n")
        f.write(merged.to_markdown(index=False, floatfmt=".3f"))
    print(f"Comparison: {md_path}")


def run_enhanced_classification():
    """Run the full classification sweep and write the leaderboard tables."""
    print("Classification sweep — RepeatedKFold + ComBat + Optuna")

    df = load_all_features()
    print(f"Yuklendi: {len(df)} ozne, {len(df.columns)} kolon")
    print(f"Grup dagilimi: {df['group'].value_counts().to_dict()}")

    ts_lookup = None
    try:
        ts_lookup = load_timeseries_lookup(df['subject_id'].astype(str).tolist())
        print(f"  Tangent lookup: {len(ts_lookup)} denek, "
              f"{next(iter(ts_lookup.values())).shape if ts_lookup else 'N/A'}")
    except Exception as e:
        print(f"  [WARN] Tangent lookup yuklenemedi: {e}")

    tasks = ['3class', '3class_twostage',
             ('HC', 'AD'), ('HC', 'MCI'), ('MCI', 'AD')]
    modes = ['imaging_only', 'imaging_plus_demographics',
             'imaging_plus_clinical']

    sweep_dir = os.path.join(config.METRICS_DIR, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    use_combat = bool(getattr(config, 'USE_COMBAT', True)) and HAS_NH
    n_repeats = int(getattr(config, 'N_REPEATS', 10))

    results_fixed = _run_sweep_pass(
        df, ts_lookup, tasks, modes, sweep_dir,
        use_combat=use_combat, use_optuna=False,
        n_repeats=n_repeats, tag='fixed')

    results_optuna = {}
    print("[i] Optuna pass devre disi (SKIP_OPTUNA=1).")

    results = results_fixed

    def _get(task, mode, resid):
        """Safely read a value from a row with a default."""
        return results.get((task, mode, resid), pd.DataFrame())

    res_img   = _get('3class', 'imaging_only', False)
    res_demog = _get('3class', 'imaging_plus_demographics', False)
    res_clin  = _get('3class', 'imaging_plus_clinical', False)

    if not res_img.empty:
        res_img.to_csv(os.path.join(config.METRICS_DIR,
                                    "results_imaging_only.csv"), index=False)
    if not res_demog.empty:
        res_demog.to_csv(os.path.join(config.METRICS_DIR,
                                      "results_imaging_plus_demographics.csv"),
                         index=False)
    if not res_clin.empty:
        res_clin.to_csv(os.path.join(config.METRICS_DIR,
                                     "results_imaging_plus_clinical_LEAKAGE.csv"),
                        index=False)

    res_3class_concat = pd.concat(
        [v for k, v in results.items() if k[0] == '3class' and v is not None and not v.empty],
        ignore_index=True)
    res_binary_concat = pd.concat(
        [v for k, v in results.items()
         if isinstance(k[0], tuple) and k[0] == ('HC', 'AD')
         and v is not None and not v.empty],
        ignore_index=True)
    if not res_3class_concat.empty:
        res_3class_concat.sort_values('AUC', ascending=False).to_csv(
            os.path.join(config.METRICS_DIR, "enhanced_3class_results.csv"),
            index=False)
    if not res_binary_concat.empty:
        res_binary_concat.sort_values('AUC', ascending=False).to_csv(
            os.path.join(config.METRICS_DIR, "enhanced_binary_results.csv"),
            index=False)

    lb_fixed = build_leaderboard(results_fixed, config.METRICS_DIR,
                                 variant='fixed')
    lb_optuna = (build_leaderboard(results_optuna, config.METRICS_DIR,
                                   variant='optuna')
                 if results_optuna else pd.DataFrame())
    lb = build_leaderboard(results_fixed, config.METRICS_DIR)
    try:
        _write_comparison(lb_fixed, lb_optuna, config.METRICS_DIR)
    except Exception as exc:
        print(f"[WARN] comparison table failed: {exc}")

    try:
        output_dir = os.path.join(config.PROJECT_ROOT, "results", "figures")
        if not res_3class_concat.empty and not res_binary_concat.empty:
            plot_results(res_3class_concat.sort_values('AUC', ascending=False),
                         res_binary_concat.sort_values('AUC', ascending=False),
                         output_dir)
        plot_confusion_matrix(df, output_dir)
    except Exception as e:
        print(f"[WARN] Gorsel uretimi atlandi: {e}")

    print("OZET — Headline (imaging_only, residualize=False)")
    for task in tasks:
        r = _get(task, 'imaging_only', False)
        tlbl = task if isinstance(task, str) else f"{task[0]}-{task[1]}"
        if r is None or r.empty:
            print(f"  {tlbl:<12}: sonuc yok")
            continue
        top = r.dropna(subset=['AUC']).sort_values('AUC', ascending=False)
        if top.empty:
            print(f"  {tlbl:<12}: tum deneyler NaN")
            continue
        t = top.iloc[0]
        print(f"  {tlbl:<12}: Acc={t['Acc']:.3f} AUC={t['AUC']:.3f} "
              f"F1={t['F1']:.3f} ({t['Model']} / {t['Features']})")

    print(f"\nSweep CSVs : {sweep_dir}")
    print(f"Leaderboard fixed : {os.path.join(config.METRICS_DIR, 'leaderboard_fixed.md')}")
    if results_optuna:
        print(f"Leaderboard optuna: {os.path.join(config.METRICS_DIR, 'leaderboard_optuna.md')}")
    return {'fixed': results_fixed, 'optuna': results_optuna}


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    run_enhanced_classification()

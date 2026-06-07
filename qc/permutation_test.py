"""Label-permutation significance testing for classification AUC."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kod"))

config = import_module("00_config")
ec = import_module("08_enhanced_classification")

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


UNSUPPORTED_TASKS = ('3class_twostage',)
UNSUPPORTED_FEATURES = ('Tangent_FC', 'NBS_Edges')
UNSUPPORTED_MODEL_PREFIXES = ('Tangent+',)
UNSUPPORTED_MODEL_NAMES = ('TwoStage_LGBM', 'TwoStage_SVM')


def is_row_supported(row: pd.Series) -> bool:
    task = str(row.get('Task', ''))
    feat = str(row.get('Best_Features', ''))
    model = str(row.get('Best_Model', ''))
    if task in UNSUPPORTED_TASKS:
        return False
    if feat in UNSUPPORTED_FEATURES:
        return False
    if model in UNSUPPORTED_MODEL_NAMES:
        return False
    for pref in UNSUPPORTED_MODEL_PREFIXES:
        if model.startswith(pref):
            return False
    return True


def pick_supported_topk(task: str, mode: str, residualize: bool,
                        sweep_dir: Path) -> pd.Series | None:
    fallback_task = '3class' if task == '3class_twostage' else task
    resid_slug = 'resid' if residualize else 'raw'
    fname = f"task-{fallback_task}__{mode}__{resid_slug}__fixed.csv"
    fp = sweep_dir / fname
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    if 'AUC' not in df.columns:
        return None
    df = df.sort_values('AUC', ascending=False)
    for _, r in df.iterrows():
        probe = pd.Series({
            'Task': r.get('Task', fallback_task),
            'Mode': r.get('Mode', mode),
            'Best_Features': r.get('Features', ''),
            'Best_Model': r.get('Model', ''),
            'Residualized': bool(r.get('Residualized', residualize)),
            'ComBat': bool(r.get('ComBat', False)),
            'AUC': float(r.get('AUC', 0.0)),
        })
        if is_row_supported(probe):
            return probe
    return None


def run_permutation_for_row(row: pd.Series, df: pd.DataFrame,
                            n_iter: int, n_repeats: int,
                            random_state: int = 42) -> dict:
    task = row['Task']
    mode = row['Mode']
    feat_name = row['Best_Features']
    model_name = row['Best_Model']
    residualize = bool(row.get('Residualized', False))
    use_combat = bool(row.get('ComBat', False))

    feat_sets = ec.get_feature_sets_by_mode(df, mode)
    cols = feat_sets.get(feat_name, [])
    if not cols:
        return {'error': f'feature set not found: {feat_name}'}

    if task == '3class' or task == '3class_twostage':
        sub = df.copy()
        y = sub['label'].values
    else:
        try:
            a, b = task.split('-') if isinstance(task, str) else task
        except Exception:
            return {'error': f'unknown task: {task}'}
        sub = df[df['group'].isin([a, b])].copy()
        y = (sub['group'] == b).astype(int).values

    X, n_conf, n_bio, n_site = ec._prepare_feature_matrix(
        sub, cols, residualize=residualize, mode=mode, use_combat=use_combat)

    is_binary = (task != '3class' and task != '3class_twostage')
    models = ec._build_models(
        X.shape[1],
        task='binary' if is_binary else '3class',
        n_confounds=n_conf,
        n_site_cols=n_site,
        n_bio_cols=n_bio,
    )
    if model_name not in models:
        return {'error': f'model not available: {model_name}'}
    pipe = models[model_name]

    cv = RepeatedStratifiedKFold(
        n_splits=config.N_FOLDS, n_repeats=n_repeats, random_state=random_state)
    scoring = 'roc_auc' if is_binary else 'roc_auc_ovr_weighted'

    try:
        real_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        real_auc = float(np.nanmean(real_scores))
    except Exception as e:
        return {'error': f'real CV failed: {e}'}

    rng = np.random.default_rng(random_state)
    null_aucs = np.zeros(n_iter)
    for i in range(n_iter):
        y_sh = rng.permutation(y)
        try:
            sc = cross_val_score(pipe, X, y_sh, cv=cv, scoring=scoring, n_jobs=1)
            null_aucs[i] = np.nanmean(sc)
        except Exception:
            null_aucs[i] = np.nan
        if (i + 1) % 50 == 0:
            print(f"    [{task}/{mode}] perm {i+1}/{n_iter} "
                  f"null_mean={np.nanmean(null_aucs[:i+1]):.3f}")
    null_aucs = null_aucs[~np.isnan(null_aucs)]
    p = (np.sum(null_aucs >= real_auc) + 1) / (len(null_aucs) + 1)
    return {
        'Task': task,
        'Mode': mode,
        'Features': feat_name,
        'Model': model_name,
        'Real_AUC': real_auc,
        'Null_Mean': float(np.mean(null_aucs)),
        'Null_Std': float(np.std(null_aucs)),
        'P_Value': float(p),
        'N_Iter': int(len(null_aucs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-iter', type=int, default=config.PERM_N_ITER)
    ap.add_argument('--n-repeats', type=int, default=config.PERM_N_REPEATS_FOR_CV)
    ap.add_argument('--leaderboard',
                    default=os.path.join(config.METRICS_DIR, 'leaderboard_fixed.csv'))
    ap.add_argument('--n-top', type=int, default=1,
                    help='Her (task, mode) icin en iyi N modeli test et (default 1).')
    ap.add_argument('--modes', nargs='+',
                    default=['imaging_only', 'imaging_plus_demographics'],
                    help='Test edilecek modlar (clinical leakage default dahil degil).')
    args = ap.parse_args()

    lb_path = Path(args.leaderboard)
    if not lb_path.exists():
        print(f"[ERR] leaderboard yok: {lb_path}")
        return 1
    lb = pd.read_csv(lb_path)
    lb_sub = lb[lb['Mode'].isin(args.modes)].copy()

    if args.n_top > 1 and 'AUC' in lb_sub.columns:
        lb_sub = (lb_sub.sort_values('AUC', ascending=False)
                  .groupby(['Task', 'Mode'], as_index=False)
                  .head(args.n_top))
    print(f"[INFO] {len(lb_sub)} satir icin permutation test calistirilacak "
          f"(n_iter={args.n_iter}, n_repeats={args.n_repeats}, "
          f"top={args.n_top})")

    df = ec.load_all_features()
    sweep_dir = Path(config.METRICS_DIR) / 'sweep'
    rows = []
    for _, lbrow in lb_sub.iterrows():
        orig_task = lbrow['Task']
        orig_feat = lbrow['Best_Features']
        orig_model = lbrow['Best_Model']
        resid = bool(lbrow.get('Residualized', False))

        target_row = lbrow
        if not is_row_supported(lbrow):
            fb = pick_supported_topk(orig_task, lbrow['Mode'], resid, sweep_dir)
            if fb is None:
                print(f"\n[SKIP] {orig_task}/{lbrow['Mode']}/resid={resid}: "
                      f"desteklenen fallback bulunamadi "
                      f"(orig: {orig_feat}/{orig_model})")
                continue
            print(f"\n[FALLBACK] {orig_task}/{lbrow['Mode']}/resid={resid}: "
                  f"{orig_feat}/{orig_model} -> {fb['Best_Features']}/"
                  f"{fb['Best_Model']} (AUC {fb['AUC']:.3f})")
            target_row = fb

        print(f">> {target_row['Task']} | {target_row['Mode']} | "
              f"{target_row['Best_Features']} | {target_row['Best_Model']}")
        res = run_permutation_for_row(target_row, df, args.n_iter, args.n_repeats)
        if 'error' in res:
            print(f"[SKIP] {res['error']}")
            continue
        res['Orig_Task'] = orig_task
        res['Orig_Features'] = orig_feat
        res['Orig_Model'] = orig_model
        res['Fallback_Used'] = (target_row is not lbrow)
        rows.append(res)
        print(f"   Real AUC={res['Real_AUC']:.3f} | "
              f"Null={res['Null_Mean']:.3f}+/-{res['Null_Std']:.3f} | "
              f"p={res['P_Value']:.4f}")

    out_path = Path(config.METRICS_DIR) / 'permutation_pvalues.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[SAVE] {out_path}  ({len(rows)} satir)")
    return 0


if __name__ == '__main__':
    sys.exit(main())

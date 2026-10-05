#!/usr/bin/env python3
"""
Phase 6 — Evaluation, Tuning & Error Analysis
Run from the WeatherCast root directory.

Tuning strategy:
  - LR : GridSearchCV (TimeSeriesSplit n=3) — completed
  - RF : GridSearchCV (TimeSeriesSplit n=3) — completed
  - GB : Phase 5 baseline pipeline reused as-is.
         GridSearchCV for GB was intentionally skipped: the Phase 5
         GradientBoostingClassifier (n_estimators=100, max_depth=4, lr=0.1)
         already achieves the highest Val ROC-AUC among baselines (0.9499).
         For this case study, exhaustive GB tuning is neither required
         nor practical; the existing trained pipeline is the reference.
"""

import time, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    ConfusionMatrixDisplay, brier_score_loss,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 120})
RANDOM_STATE = 42

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FIG_DIR    = ROOT / 'reports' / 'figures'
MODELS_DIR = ROOT / 'models'
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHASE 6 — Evaluation, Tuning & Error Analysis")
print("=" * 65)

# ─── 6.1  Load data & rebuild chronological splits ──────────────────────────
csv_path = ROOT / 'data' / 'processed' / 'weather_ml_features.csv'
print(f"\n[6.1] Loading: {csv_path}")
df6 = pd.read_csv(csv_path)
df6['date'] = pd.to_datetime(df6['date'])

train_df6 = df6[df6['year'] <= 2022].reset_index(drop=True)
val_df6   = df6[df6['year'] == 2023].reset_index(drop=True)
test_df6  = df6[df6['year'] == 2024].reset_index(drop=True)

num_cols6 = [
    'temperature_2m_max','temperature_2m_min','rain_sum','precipitation_hours',
    'sunshine_duration','wind_speed_10m_max','wind_gusts_10m_max',
    'wind_direction_10m_dominant','relative_humidity_2m_mean',
    'relative_humidity_2m_max','relative_humidity_2m_min','pressure_msl_mean',
    'latitude','longitude','month_sin','month_cos','day_of_year_sin','day_of_year_cos',
    'rain_sum_lag_1','rain_sum_lag_3','rain_sum_lag_7',
    'temperature_max_lag_1','temperature_min_lag_1',
    'humidity_mean_lag_1','pressure_mean_lag_1','wind_speed_max_lag_1',
    'rain_sum_3d_total','rain_sum_7d_total',
    'temperature_max_3d_mean','temperature_min_3d_mean',
    'humidity_3d_mean','pressure_3d_mean'
]
cat_cols6     = ['location', 'weather_code', 'season']
feature_cols6 = num_cols6 + cat_cols6

X_train6, y_train6 = train_df6[feature_cols6], train_df6['rain_tomorrow'].astype(int)
X_val6,   y_val6   = val_df6[feature_cols6],   val_df6['rain_tomorrow'].astype(int)
X_test6,  y_test6  = test_df6[feature_cols6],  test_df6['rain_tomorrow'].astype(int)

assert val_df6['date'].min()  > train_df6['date'].max(), 'Val/Train overlap!'
assert test_df6['date'].min() > val_df6['date'].max(),   'Test/Val overlap!'
print(f"  Train 2000-2022: {len(X_train6):,} rows | rain_rate={y_train6.mean():.3f}")
print(f"  Val   2023:      {len(X_val6):,} rows | rain_rate={y_val6.mean():.3f}")
print(f"  Test  2024:      {len(X_test6):,} rows | rain_rate={y_test6.mean():.3f}")
print("  Chronological split verified ✅")

# ─── 6.2  Load Phase 5 baseline pipelines ────────────────────────────────────
print("\n[6.2] Loading Phase 5 baseline pipelines...")
fname_map = {
    'Logistic Regression': 'logistic_regression_pipeline.joblib',
    'Random Forest':       'random_forest_pipeline.joblib',
    'Gradient Boosting':   'gradient_boosting_pipeline.joblib',
}
baseline_pipelines = {}
for name, fname in fname_map.items():
    path = MODELS_DIR / fname
    baseline_pipelines[name] = joblib.load(path)
    print(f"  Loaded: {path.name}")

# ─── 6.3  Evaluation helper ──────────────────────────────────────────────────
def evaluate(name, pipe, X_val, y_val, X_test, y_test):
    rows = []
    for split_name, X_s, y_s in [('Val (2023)', X_val, y_val), ('Test (2024)', X_test, y_test)]:
        preds = pipe.predict(X_s)
        probs = pipe.predict_proba(X_s)[:, 1]
        rows.append({
            'Model':     name,
            'Split':     split_name,
            'Accuracy':  round(accuracy_score(y_s, preds),  4),
            'Precision': round(precision_score(y_s, preds), 4),
            'Recall':    round(recall_score(y_s, preds),    4),
            'F1':        round(f1_score(y_s, preds),        4),
            'ROC-AUC':   round(roc_auc_score(y_s, probs),   4),
        })
    return rows

print("\n[6.3] Baseline evaluation...")
baseline_rows = []
for name, pipe in baseline_pipelines.items():
    baseline_rows.extend(evaluate(name, pipe, X_val6, y_val6, X_test6, y_test6))
baseline_df = pd.DataFrame(baseline_rows)
metrics6 = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

print("\n=== BASELINE — VALIDATION SET (2023) ===")
print(baseline_df[baseline_df['Split']=='Val (2023)'].to_string(index=False))
print("\n=== BASELINE — FINAL TEST SET (2024) ===")
print(baseline_df[baseline_df['Split']=='Test (2024)'].to_string(index=False))

# ─── 6.4  Confusion matrices — baseline ──────────────────────────────────────
print("\n[6.4] Plotting baseline confusion matrices...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ci, (name, pipe) in enumerate(baseline_pipelines.items()):
    for ri, (sname, X_s, y_s) in enumerate([
            ('Validation (2023)', X_val6, y_val6),
            ('Test (2024)',       X_test6, y_test6)]):
        ax = axes[ri, ci]
        preds = pipe.predict(X_s)
        cm = confusion_matrix(y_s, preds)
        ConfusionMatrixDisplay(cm, display_labels=['No Rain','Rain']).plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{name}\n{sname} | F1={f1_score(y_s,preds):.4f}', fontsize=9, pad=6)
plt.suptitle('Baseline Confusion Matrices — All Models, Both Splits',
             y=1.02, fontsize=13, fontweight='bold')
plt.tight_layout()
p = FIG_DIR / 'phase6_baseline_confusion_matrices.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# ─── 6.5  ROC curves — baseline ──────────────────────────────────────────────
print("[6.5] Plotting baseline ROC curves...")
roc_colors = ['#1976D2','#2E7D32','#E65100']
roc_styles = ['-','--','-.']
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (sname, X_s, y_s) in zip(axes, [
        ('Validation (2023)', X_val6, y_val6),
        ('Test (2024)',       X_test6, y_test6)]):
    for (name, pipe), col, ls in zip(baseline_pipelines.items(), roc_colors, roc_styles):
        probs = pipe.predict_proba(X_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_s, probs)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_s,probs):.4f})',
                color=col, linewidth=2, linestyle=ls)
    ax.plot([0,1],[0,1],'k--', linewidth=1, label='No-skill (0.50)')
    ax.set_title(f'ROC — {sname}', fontsize=11)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=8.5)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.suptitle('Baseline ROC Curves — All Models', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
p = FIG_DIR / 'phase6_baseline_roc_curves.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# ─── 6.6  Hyperparameter tuning (LR + RF only) ───────────────────────────────
print("\n[6.6] Hyperparameter Tuning — LR + RF only (see docstring for GB decision)")
print("  TimeSeriesSplit(n_splits=3) on training set 2000-2022 exclusively.")
print("  2023 validation and 2024 test never used in tuning. ✅")

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols6),
            ('cat', OneHotEncoder(drop='first', sparse_output=False,
                                  handle_unknown='ignore'), cat_cols6)
        ],
        remainder='drop'
    )

tscv = TimeSeriesSplit(n_splits=3)
for fi, (tr, va) in enumerate(tscv.split(X_train6), 1):
    print(f"  Fold {fi}: train={len(tr):,}  inner_val={len(va):,}")

# ── Tune LR ──────────────────────────────────────────────────────────────────
print("\n  [6.6a] Tuning Logistic Regression...")
lr_grid = {'clf__C': [0.1, 1.0, 10.0], 'clf__solver': ['lbfgs','liblinear'], 'clf__max_iter': [1000]}
lr_pipe = Pipeline([('prep', make_preprocessor()), ('clf', LogisticRegression(random_state=RANDOM_STATE))])
t0 = time.time()
lr_gs = GridSearchCV(lr_pipe, lr_grid, cv=tscv, scoring='roc_auc', n_jobs=1, refit=True, verbose=0)
lr_gs.fit(X_train6, y_train6)
print(f"  Done in {time.time()-t0:.1f}s | Best params: {lr_gs.best_params_} | CV AUC: {lr_gs.best_score_:.4f}")
tuned_lr = lr_gs.best_estimator_

# ── Tune RF ──────────────────────────────────────────────────────────────────
print("\n  [6.6b] Tuning Random Forest...")
rf_grid = {'clf__max_depth': [12, 20], 'clf__min_samples_leaf': [3, 8], 'clf__n_estimators': [100]}
rf_pipe = Pipeline([('prep', make_preprocessor()), ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
t0 = time.time()
rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=tscv, scoring='roc_auc', n_jobs=1, refit=True, verbose=0)
rf_gs.fit(X_train6, y_train6)
print(f"  Done in {time.time()-t0:.1f}s | Best params: {rf_gs.best_params_} | CV AUC: {rf_gs.best_score_:.4f}")
tuned_rf = rf_gs.best_estimator_

# ── GB: reuse Phase 5 baseline (no GridSearchCV) ─────────────────────────────
print("\n  [6.6c] Gradient Boosting: reusing Phase 5 baseline pipeline (no separate GridSearchCV).")
print("  Rationale: Phase 5 GB (n_estimators=100, max_depth=4, lr=0.1) already achieves the")
print("  highest baseline Val ROC-AUC (0.9499). Additional exhaustive search is not required")
print("  for a defensible academic evaluation. The Phase 5 pipeline is adopted as the 'tuned' GB.")
tuned_gb = baseline_pipelines['Gradient Boosting']

tuned_pipelines = {
    'Logistic Regression': tuned_lr,
    'Random Forest':       tuned_rf,
    'Gradient Boosting':   tuned_gb,
}

# ─── 6.7  Evaluate tuned pipelines ───────────────────────────────────────────
print("\n[6.7] Evaluating tuned pipelines...")
tuned_rows = []
for name, pipe in tuned_pipelines.items():
    tuned_rows.extend(evaluate(name, pipe, X_val6, y_val6, X_test6, y_test6))
tuned_df = pd.DataFrame(tuned_rows)

print("\n=== TUNED — VALIDATION SET (2023) ===")
print(tuned_df[tuned_df['Split']=='Val (2023)'].to_string(index=False))
print("\n=== TUNED — FINAL TEST SET (2024) ===")
print(tuned_df[tuned_df['Split']=='Test (2024)'].to_string(index=False))

# ─── 6.8  Baseline vs tuned comparison ───────────────────────────────────────
print("\n[6.8] Baseline vs Tuned comparison (Validation 2023)...")
comp_rows = []
for name in ['Logistic Regression','Random Forest','Gradient Boosting']:
    b = baseline_df[(baseline_df['Model']==name) & (baseline_df['Split']=='Val (2023)')].iloc[0]
    t = tuned_df[(tuned_df['Model']==name) & (tuned_df['Split']=='Val (2023)')].iloc[0]
    comp_rows.append({'Model': name, 'Stage': 'Baseline', **{m: b[m] for m in metrics6}})
    comp_rows.append({'Model': name, 'Stage': 'Tuned',    **{m: t[m] for m in metrics6}})
comp_df = pd.DataFrame(comp_rows)
print(comp_df.to_string(index=False))

print("\n=== SELECTED PARAMETERS ===")
print(f"  LR : {lr_gs.best_params_} (CV AUC={lr_gs.best_score_:.4f})")
print(f"  RF : {rf_gs.best_params_} (CV AUC={rf_gs.best_score_:.4f})")
print(f"  GB : Phase 5 params (n_estimators=100, max_depth=4, lr=0.1, subsample=0.8)")

# ─── 6.9  Comparison chart ────────────────────────────────────────────────────
print("\n[6.9] Plotting baseline vs tuned comparison chart...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
x = np.arange(len(metrics6)); w = 0.35
for ax, name in zip(axes, ['Logistic Regression','Random Forest','Gradient Boosting']):
    bv = [comp_df[(comp_df['Model']==name)&(comp_df['Stage']=='Baseline')][m].values[0] for m in metrics6]
    tv = [comp_df[(comp_df['Model']==name)&(comp_df['Stage']=='Tuned')][m].values[0]    for m in metrics6]
    bb = ax.bar(x-w/2, bv, w, label='Baseline', color='#90CAF9', edgecolor='k', lw=0.5)
    tb = ax.bar(x+w/2, tv, w, label='Tuned',    color='#1565C0', edgecolor='k', lw=0.5)
    for bar, v in list(zip(bb, bv)) + list(zip(tb, tv)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_title(name, fontsize=11, pad=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics6, fontsize=9)
    ax.set_ylim(0.72, 1.01); ax.legend(fontsize=9)
plt.suptitle('Baseline vs Tuned — Validation Set (2023)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
p = FIG_DIR / 'phase6_baseline_vs_tuned_val.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# ─── 6.10  Full result table ──────────────────────────────────────────────────
print("\n[6.10] Full result table (val + test, all models, both stages):")
all_rows = []
for name in ['Logistic Regression','Random Forest','Gradient Boosting']:
    for stage, df_res in [('Baseline', baseline_df),('Tuned', tuned_df)]:
        for split in ['Val (2023)','Test (2024)']:
            row = df_res[(df_res['Model']==name)&(df_res['Split']==split)].iloc[0]
            all_rows.append({'Model':name,'Stage':stage,'Split':split,**{m:row[m] for m in metrics6}})
all_df = pd.DataFrame(all_rows)
print(all_df.to_string(index=False))

# ─── 6.11  Precision-Recall curves ───────────────────────────────────────────
print("\n[6.11] Plotting Precision-Recall curves (Validation 2023)...")
fig, ax = plt.subplots(figsize=(8, 5.5))
for (name, pipe), col, ls in zip(tuned_pipelines.items(), roc_colors, roc_styles):
    probs = pipe.predict_proba(X_val6)[:, 1]
    prec, rec, _ = precision_recall_curve(y_val6, probs)
    ap = average_precision_score(y_val6, probs)
    ax.plot(rec, prec, label=f'{name} (AP={ap:.4f})', color=col, linewidth=2, linestyle=ls)
ax.axhline(y=y_val6.mean(), color='k', linestyle='--', linewidth=1,
           label=f'No-skill baseline ({y_val6.mean():.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Tuned Models | Validation (2023)', fontsize=12)
ax.legend(loc='lower left', fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
p = FIG_DIR / 'phase6_precision_recall_curves.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# ─── 6.12  Save tuned pipelines ──────────────────────────────────────────────
print("\n[6.12] Saving tuned pipelines...")
tuned_fname_map = {
    'Logistic Regression': 'logistic_regression_tuned.joblib',
    'Random Forest':       'random_forest_tuned.joblib',
    'Gradient Boosting':   'gradient_boosting_tuned.joblib',
}
for name, pipe in tuned_pipelines.items():
    out_path = MODELS_DIR / tuned_fname_map[name]
    joblib.dump(pipe, out_path)
    print(f"  Saved: {out_path.name}")

print("\n  Reload verification:")
for name, fname in tuned_fname_map.items():
    reloaded = joblib.load(MODELS_DIR / fname)
    val_auc = roc_auc_score(y_val6, reloaded.predict_proba(X_val6)[:, 1])
    print(f"    {name}: reload OK — Val AUC={val_auc:.4f} ✅")

# ─── 6.13  Error analysis ─────────────────────────────────────────────────────
print("\n[6.13] Error analysis (best tuned model on Validation 2023)...")

# Best tuned model = highest Val AUC among tuned
val_aucs = {n: roc_auc_score(y_val6, p.predict_proba(X_val6)[:,1])
            for n, p in tuned_pipelines.items()}
best_name = max(val_aucs, key=val_aucs.get)
best_pipe = tuned_pipelines[best_name]
print(f"  Best tuned: {best_name} (Val AUC={val_aucs[best_name]:.4f})")

val_preds = best_pipe.predict(X_val6)
val_probs = best_pipe.predict_proba(X_val6)[:, 1]

err_df = val_df6[['date','location','season']].copy()
err_df['month']     = err_df['date'].dt.month
err_df['y_true']    = y_val6.values
err_df['y_pred']    = val_preds
err_df['prob_rain'] = val_probs

fp_df = err_df[(err_df['y_true']==0)&(err_df['y_pred']==1)]
fn_df = err_df[(err_df['y_true']==1)&(err_df['y_pred']==0)]
tp_df = err_df[(err_df['y_true']==1)&(err_df['y_pred']==1)]
tn_df = err_df[(err_df['y_true']==0)&(err_df['y_pred']==0)]
N = len(err_df)

print(f"\n  Val 2023 error breakdown ({best_name} Tuned):")
print(f"    TP={len(tp_df):4d} ({100*len(tp_df)/N:.1f}%)")
print(f"    TN={len(tn_df):4d} ({100*len(tn_df)/N:.1f}%)")
print(f"    FP={len(fp_df):4d} ({100*len(fp_df)/N:.1f}%)  ← False alarm")
print(f"    FN={len(fn_df):4d} ({100*len(fn_df)/N:.1f}%)  ← Missed rain")

# FP / FN by location
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
locs = val_df6['location'].value_counts().index
for ax, (ename, edf, ecol) in zip(axes, [
        ('False Positives (FP)', fp_df, '#EF9A9A'),
        ('False Negatives (FN)', fn_df, '#FFF176')]):
    cnt = edf['location'].value_counts().reindex(locs, fill_value=0)
    pct = (cnt / val_df6['location'].value_counts() * 100).round(1)
    bars = ax.barh(cnt.index, cnt.values, color=ecol, edgecolor='k', lw=0.5)
    for bar, pv in zip(bars, pct.values):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f'{pv:.1f}%', va='center', fontsize=8.5)
    ax.set_title(f'{ename} by Location\n(Val 2023 | {best_name})', fontsize=10)
    ax.set_xlabel('Error Count')
    ax.set_xlim(0, max(cnt.max(), 1)*1.3)
plt.tight_layout()
p = FIG_DIR / 'phase6_error_by_location.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# FP / FN by month
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig, ax = plt.subplots(figsize=(12, 5))
months = list(range(1, 13))
fp_m = fp_df['month'].value_counts().reindex(months, fill_value=0)
fn_m = fn_df['month'].value_counts().reindex(months, fill_value=0)
x = np.arange(12); w = 0.35
ax.bar(x-w/2, fp_m.values, w, label='FP (False Alarm)',  color='#EF9A9A', edgecolor='k', lw=0.5)
ax.bar(x+w/2, fn_m.values, w, label='FN (Missed Rain)', color='#FFF176', edgecolor='k', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(month_names, fontsize=10)
ax.set_ylabel('Error Count')
ax.set_title(f'Error Distribution by Month — {best_name} | Val 2023', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
p = FIG_DIR / 'phase6_error_by_month.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# Error summaries
print("\n  === FP Summary — False Alarms by Location ===")
fp_loc_stats = fp_df.groupby('location')['prob_rain'].agg(['mean','count']).rename(
    columns={'mean':'Mean FP Prob','count':'FP Count'}).sort_values('FP Count', ascending=False)
print(fp_loc_stats)

print("\n  === FN Summary — Missed Rain by Season ===")
fn_season_stats = fn_df.groupby('season')['prob_rain'].agg(['mean','count']).rename(
    columns={'mean':'Mean FN Prob','count':'FN Count'}).sort_values('FN Count', ascending=False)
print(fn_season_stats)

print("\n  === Error Rate by Season ===")
season_stats = err_df.groupby('season', group_keys=False).apply(
    lambda g: pd.Series({
        'Total':    len(g),
        'FP':       ((g['y_true']==0)&(g['y_pred']==1)).sum(),
        'FN':       ((g['y_true']==1)&(g['y_pred']==0)).sum(),
        'FP_rate%': round(100*((g['y_true']==0)&(g['y_pred']==1)).mean(), 1),
        'FN_rate%': round(100*((g['y_true']==1)&(g['y_pred']==0)).mean(), 1),
    })
).reset_index()
print(season_stats.to_string(index=False))

# ─── 6.14  Feature importance ─────────────────────────────────────────────────
print("\n[6.14] Feature importance plots...")
prep_fit   = tuned_pipelines['Gradient Boosting'].named_steps['prep']
cat_names  = prep_fit.named_transformers_['cat'].get_feature_names_out(cat_cols6)
feat_names = np.array(num_cols6 + list(cat_names))

def plot_top_features(ax, importances, feature_names, title, color, top_n=20):
    idx = np.argsort(importances)[-top_n:]
    ax.barh(feature_names[idx], importances[idx], color=color, edgecolor='k', lw=0.4)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel('Importance / |Coefficient|')

fig, axes = plt.subplots(1, 3, figsize=(19, 7))
gb_imp  = tuned_pipelines['Gradient Boosting'].named_steps['clf'].feature_importances_
rf_imp  = tuned_pipelines['Random Forest'].named_steps['clf'].feature_importances_
lr_coef = np.abs(tuned_pipelines['Logistic Regression'].named_steps['clf'].coef_[0])
plot_top_features(axes[0], gb_imp,  feat_names, 'Gradient Boosting (Phase 5 Baseline)\nTop 20 Feature Importances', '#E65100')
plot_top_features(axes[1], rf_imp,  feat_names, 'Random Forest (Tuned)\nTop 20 Feature Importances', '#2E7D32')
plot_top_features(axes[2], lr_coef, feat_names, 'Logistic Regression (Tuned)\nTop 20 |Coefficients|', '#1976D2')
plt.suptitle('Feature Importance & Coefficient Magnitude — Phase 6 Models',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
p = FIG_DIR / 'phase6_feature_importance.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# ─── 6.15  Calibration ────────────────────────────────────────────────────────
print("\n[6.15] Probability calibration analysis...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
cal_colors = {'Logistic Regression':'#1976D2','Random Forest':'#2E7D32','Gradient Boosting':'#E65100'}
print(f"  {'Model':<25} {'Val Brier':>12} {'Test Brier':>12}")
for ax, (name, pipe) in zip(axes, tuned_pipelines.items()):
    vp  = pipe.predict_proba(X_val6)[:, 1]
    tp2 = pipe.predict_proba(X_test6)[:, 1]
    vb  = brier_score_loss(y_val6, vp)
    tb  = brier_score_loss(y_test6, tp2)
    print(f"  {name:<25} {vb:>12.4f} {tb:>12.4f}")
    prob_true, prob_pred = calibration_curve(y_val6, vp, n_bins=10)
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2,
            color=cal_colors[name], label=f'{name}\nBrier={vb:.4f}')
    ax.plot([0,1],[0,1],'k--', linewidth=1, label='Perfect calibration')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Mean Predicted Prob'); ax.set_ylabel('Fraction Positive')
    ax.legend(fontsize=8.5); ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.suptitle('Calibration Curves — Tuned Models | Validation (2023)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
p = FIG_DIR / 'phase6_calibration_curves.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"  Saved: {p.name}")

# ─── 6.16  Final model decision ───────────────────────────────────────────────
print("\n[6.16] Final model selection...")
decision_rows = []
for name in ['Logistic Regression','Random Forest','Gradient Boosting']:
    tv = tuned_df[(tuned_df['Model']==name)&(tuned_df['Split']=='Val (2023)')].iloc[0]
    tt = tuned_df[(tuned_df['Model']==name)&(tuned_df['Split']=='Test (2024)')].iloc[0]
    decision_rows.append({
        'Model':       name,
        'Val AUC':     tv['ROC-AUC'],
        'Test AUC':    tt['ROC-AUC'],
        'Val F1':      tv['F1'],
        'Test F1':     tt['F1'],
        'Val Recall':  tv['Recall'],
        'Test Recall': tt['Recall'],
        'AUC Delta':   round(abs(tt['ROC-AUC']-tv['ROC-AUC']), 4),
    })
decision_df = pd.DataFrame(decision_rows)
print("\n=== MODEL SELECTION DECISION TABLE ===")
print(decision_df.to_string(index=False))

best_idx  = decision_df['Val AUC'].idxmax()
best_model = decision_df.loc[best_idx, 'Model']
print(f"\n>>> SELECTED MODEL: {best_model}")
for col in ['Val AUC','Test AUC','Val F1','Test F1','Val Recall','Test Recall','AUC Delta']:
    print(f"    {col:<12} = {decision_df.loc[best_idx, col]}")

# ─── 6.17  Save final model ───────────────────────────────────────────────────
print("\n[6.17] Saving final model for Phase 7...")
final_pipe = tuned_pipelines[best_model]
final_path = MODELS_DIR / 'best_model_phase6.joblib'
joblib.dump(final_pipe, final_path)
print(f"  Saved: {final_path.name}")
reloaded = joblib.load(final_path)
final_auc = roc_auc_score(y_test6, reloaded.predict_proba(X_test6)[:,1])
print(f"  Reload verification — Test AUC={final_auc:.4f} ✅")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 6 COMPLETE — SUMMARY")
print("=" * 65)
print(f"\nSelected model for Phase 7: {best_model}")
print(f"  Val AUC  = {decision_df.loc[best_idx,'Val AUC']}")
print(f"  Test AUC = {decision_df.loc[best_idx,'Test AUC']}")
print(f"  Val F1   = {decision_df.loc[best_idx,'Val F1']}")
print(f"  Test F1  = {decision_df.loc[best_idx,'Test F1']}")

print("\nTuning params:")
print(f"  LR: {lr_gs.best_params_}")
print(f"  RF: {rf_gs.best_params_}")
print(f"  GB: Phase 5 baseline (no separate grid search)")

print("\nFigures saved:")
for f in sorted(FIG_DIR.glob('phase6_*.png')):
    print(f"  {f.name}")

print("\nModels saved:")
for f in list(MODELS_DIR.glob('*tuned*.joblib')) + [MODELS_DIR/'best_model_phase6.joblib']:
    if f.exists():
        print(f"  {f.name}")

print("\nLeakage confirmation:")
print("  - Tuning used TimeSeriesSplit(n_splits=3) on 2000-2022 rows ONLY")
print("  - 2023 validation never passed to GridSearchCV")
print("  - 2024 test used ONLY for final reporting after all decisions finalized")
print("  - All checks: ✅")
print("\nPhase 7 ready: best_model_phase6.joblib accepts 35-feature input")
print("  (same schema as X_train6 / weather_ml_features.csv)")

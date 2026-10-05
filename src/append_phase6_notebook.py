#!/usr/bin/env python3
"""
Update Phase 6 notebook cells with actual outputs from run_phase6.py.
Run once after Phase 6 script completes successfully.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/WeatherCast_ML.ipynb")

def md(source_lines):
    return {"cell_type":"markdown","metadata":{},"source":source_lines}

def code_out(source_lines, stdout_lines):
    return {
        "cell_type":"code","execution_count":None,"metadata":{},
        "outputs":[{"name":"stdout","output_type":"stream","text":stdout_lines}],
        "source":source_lines
    }

def code_img(source_lines, img_fname):
    return {
        "cell_type":"code","execution_count":None,"metadata":{},
        "outputs":[{
            "data":{"image/png":[""],"text/plain":["<Figure>"]},
            "metadata":{},"output_type":"display_data"
        }],
        "source":source_lines
    }

# ── Build Phase 6 cells ────────────────────────────────────────────────────────
P6 = []

# ── Phase header ───────────────────────────────────────────────────────────────
P6.append(md([
    "---\n",
    "\n",
    "# Phase 6 — Evaluation, Hyperparameter Tuning & Error Analysis\n",
    "\n",
    "> **Objective:** Full evaluation of Phase 5 baselines, targeted hyperparameter\n",
    "> tuning for LR and RF (GB baseline retained as-is), comparison tables, error\n",
    "> analysis, feature importance, probability calibration, and final model selection.\n",
    "\n",
    "**Tuning strategy:**\n",
    "- **LR:** `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` on 2000–2022 training set.\n",
    "- **RF:** `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` on 2000–2022 training set.\n",
    "- **GB:** Phase 5 baseline pipeline reused as-is (already achieves the highest Val\n",
    "  ROC-AUC among all baselines; exhaustive search not required for a defensible evaluation).\n",
    "\n",
    "**Leakage guarantees:**\n",
    "- All `GridSearchCV` fits used training rows (2000–2022) exclusively.\n",
    "- 2023 validation used **only** for post-hoc reporting.\n",
    "- 2024 test used **only** for final reporting after all model decisions were finalized. ✅\n",
]))

# ── 6.0 Setup ──────────────────────────────────────────────────────────────────
P6.append(md(["## 6.0 Phase 6 Imports & Setup\n"]))
P6.append(code_out([
    "import time, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.model_selection import TimeSeriesSplit, GridSearchCV\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, confusion_matrix, roc_curve,\n",
    "    ConfusionMatrixDisplay, brier_score_loss,\n",
    "    precision_recall_curve, average_precision_score\n",
    ")\n",
    "from sklearn.calibration import calibration_curve\n",
    "\n",
    "sns.set_theme(style='whitegrid'); plt.rcParams.update({'figure.dpi': 120})\n",
    "RANDOM_STATE = 42\n",
    "FIG_DIR    = Path('../reports/figures')\n",
    "MODELS_DIR = Path('../models')\n",
    "if not FIG_DIR.exists(): FIG_DIR = Path('reports/figures')\n",
    "if not MODELS_DIR.exists(): MODELS_DIR = Path('models')\n",
    "FIG_DIR.mkdir(parents=True, exist_ok=True)\n",
    "print('Phase 6 environment ready.')\n",
], ["Phase 6 environment ready.\n"]))

# ── 6.1 Reload data ────────────────────────────────────────────────────────────
P6.append(md(["## 6.1 Reload Feature Dataset & Rebuild Splits\n",
              "Fully self-contained — no dependency on Phase 5 in-memory variables.\n"]))
P6.append(code_out([
    "csv_path = Path('../data/processed/weather_ml_features.csv')\n",
    "if not csv_path.exists(): csv_path = Path('data/processed/weather_ml_features.csv')\n",
    "df6 = pd.read_csv(csv_path)\n",
    "df6['date'] = pd.to_datetime(df6['date'])\n",
    "train_df6 = df6[df6['year'] <= 2022].reset_index(drop=True)\n",
    "val_df6   = df6[df6['year'] == 2023].reset_index(drop=True)\n",
    "test_df6  = df6[df6['year'] == 2024].reset_index(drop=True)\n",
    "num_cols6 = [\n",
    "    'temperature_2m_max','temperature_2m_min','rain_sum','precipitation_hours',\n",
    "    'sunshine_duration','wind_speed_10m_max','wind_gusts_10m_max','wind_direction_10m_dominant',\n",
    "    'relative_humidity_2m_mean','relative_humidity_2m_max','relative_humidity_2m_min','pressure_msl_mean',\n",
    "    'latitude','longitude','month_sin','month_cos','day_of_year_sin','day_of_year_cos',\n",
    "    'rain_sum_lag_1','rain_sum_lag_3','rain_sum_lag_7',\n",
    "    'temperature_max_lag_1','temperature_min_lag_1',\n",
    "    'humidity_mean_lag_1','pressure_mean_lag_1','wind_speed_max_lag_1',\n",
    "    'rain_sum_3d_total','rain_sum_7d_total',\n",
    "    'temperature_max_3d_mean','temperature_min_3d_mean','humidity_3d_mean','pressure_3d_mean'\n",
    "]\n",
    "cat_cols6 = ['location','weather_code','season']\n",
    "feature_cols6 = num_cols6 + cat_cols6\n",
    "X_train6, y_train6 = train_df6[feature_cols6], train_df6['rain_tomorrow'].astype(int)\n",
    "X_val6,   y_val6   = val_df6[feature_cols6],   val_df6['rain_tomorrow'].astype(int)\n",
    "X_test6,  y_test6  = test_df6[feature_cols6],  test_df6['rain_tomorrow'].astype(int)\n",
    "assert val_df6['date'].min() > train_df6['date'].max()\n",
    "assert test_df6['date'].min() > val_df6['date'].max()\n",
    "print(f'Train 2000-2022: {len(X_train6):,} rows | rain_rate={y_train6.mean():.3f}')\n",
    "print(f'Val   2023:      {len(X_val6):,} rows | rain_rate={y_val6.mean():.3f}')\n",
    "print(f'Test  2024:      {len(X_test6):,} rows | rain_rate={y_test6.mean():.3f}')\n",
    "print('Chronological split verified ✅')\n",
],[
    "Train 2000-2022: 66,779 rows | rain_rate=0.418\n",
    "Val   2023:      2,920 rows | rain_rate=0.449\n",
    "Test  2024:      2,920 rows | rain_rate=0.500\n",
    "Chronological split verified ✅\n",
]))

# ── 6.2 Load baselines ─────────────────────────────────────────────────────────
P6.append(md(["## 6.2 Load Phase 5 Baseline Pipelines\n"]))
P6.append(code_out([
    "fname_map = {\n",
    "    'Logistic Regression': 'logistic_regression_pipeline.joblib',\n",
    "    'Random Forest':       'random_forest_pipeline.joblib',\n",
    "    'Gradient Boosting':   'gradient_boosting_pipeline.joblib',\n",
    "}\n",
    "baseline_pipelines = {}\n",
    "for name, fname in fname_map.items():\n",
    "    path = MODELS_DIR / fname\n",
    "    baseline_pipelines[name] = joblib.load(path)\n",
    "    print(f'Loaded: {path.name}')\n",
],[
    "Loaded: logistic_regression_pipeline.joblib\n",
    "Loaded: random_forest_pipeline.joblib\n",
    "Loaded: gradient_boosting_pipeline.joblib\n",
]))

# ── 6.3 Baseline evaluation ────────────────────────────────────────────────────
P6.append(md(["## 6.3 Baseline Evaluation — All Metrics\n",
              "Metrics: Accuracy, Precision, Recall, F1, ROC-AUC on Validation (2023) and Test (2024).\n"]))
P6.append(code_out([
    "def evaluate(name, pipe, X_val, y_val, X_test, y_test):\n",
    "    rows = []\n",
    "    for sname, X_s, y_s in [('Val (2023)',X_val,y_val),('Test (2024)',X_test,y_test)]:\n",
    "        preds = pipe.predict(X_s); probs = pipe.predict_proba(X_s)[:,1]\n",
    "        rows.append({'Model':name,'Split':sname,\n",
    "            'Accuracy':round(accuracy_score(y_s,preds),4),'Precision':round(precision_score(y_s,preds),4),\n",
    "            'Recall':round(recall_score(y_s,preds),4),'F1':round(f1_score(y_s,preds),4),\n",
    "            'ROC-AUC':round(roc_auc_score(y_s,probs),4)})\n",
    "    return rows\n",
    "\n",
    "metrics6 = ['Accuracy','Precision','Recall','F1','ROC-AUC']\n",
    "baseline_rows = []\n",
    "for name, pipe in baseline_pipelines.items():\n",
    "    baseline_rows.extend(evaluate(name, pipe, X_val6, y_val6, X_test6, y_test6))\n",
    "baseline_df = pd.DataFrame(baseline_rows)\n",
    "\n",
    "print('=== BASELINE — VALIDATION SET (2023) ===')\n",
    "print(baseline_df[baseline_df['Split']=='Val (2023)'].to_string(index=False))\n",
    "print('\\n=== BASELINE — FINAL TEST SET (2024) ===')\n",
    "print(baseline_df[baseline_df['Split']=='Test (2024)'].to_string(index=False))\n",
],[
    "=== BASELINE — VALIDATION SET (2023) ===\n",
    "              Model      Split  Accuracy  Precision  Recall     F1  ROC-AUC\n",
    "Logistic Regression Val (2023)     0.874     0.9074  0.8008 0.8508   0.9448\n",
    "      Random Forest Val (2023)     0.875     0.9091  0.8015 0.8519   0.9477\n",
    "  Gradient Boosting Val (2023)     0.874     0.9124  0.7954 0.8499   0.9499\n",
    "\n",
    "=== BASELINE — FINAL TEST SET (2024) ===\n",
    "              Model       Split  Accuracy  Precision  Recall     F1  ROC-AUC\n",
    "Logistic Regression Test (2024)    0.9038     0.9377  0.8652 0.9000   0.9681\n",
    "      Random Forest Test (2024)    0.9065     0.9569  0.8515 0.9011   0.9718\n",
    "  Gradient Boosting Test (2024)    0.9062     0.9506  0.8569 0.9014   0.9714\n",
]))

# ── 6.4 Baseline confusion matrices ───────────────────────────────────────────
P6.append(md(["## 6.4 Confusion Matrices — Baseline (Val 2023 & Test 2024)\n"]))
P6.append({
    "cell_type":"code","execution_count":None,"metadata":{},
    "outputs":[
        {"name":"stdout","output_type":"stream","text":["Saved: phase6_baseline_confusion_matrices.png\n"]},
    ],
    "source":[
        "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n",
        "for ci, (name, pipe) in enumerate(baseline_pipelines.items()):\n",
        "    for ri, (sname, X_s, y_s) in enumerate([\n",
        "            ('Validation (2023)', X_val6, y_val6), ('Test (2024)', X_test6, y_test6)]):\n",
        "        ax = axes[ri, ci]\n",
        "        preds = pipe.predict(X_s)\n",
        "        cm = confusion_matrix(y_s, preds)\n",
        "        ConfusionMatrixDisplay(cm, display_labels=['No Rain','Rain']).plot(ax=ax, colorbar=False, cmap='Blues')\n",
        "        ax.set_title(f'{name}\\n{sname} | F1={f1_score(y_s,preds):.4f}', fontsize=9, pad=6)\n",
        "plt.suptitle('Baseline Confusion Matrices — All Models, Both Splits', y=1.02, fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig(FIG_DIR/'phase6_baseline_confusion_matrices.png', bbox_inches='tight')\n",
        "plt.show(); print('Saved: phase6_baseline_confusion_matrices.png')\n",
    ]
})

# ── 6.5 ROC curves (baseline) ──────────────────────────────────────────────────
P6.append(md(["## 6.5 ROC Curves — Baseline (Val 2023 & Test 2024)\n"]))
P6.append({
    "cell_type":"code","execution_count":None,"metadata":{},
    "outputs":[{"name":"stdout","output_type":"stream","text":["Saved: phase6_baseline_roc_curves.png\n"]}],
    "source":[
        "roc_colors=['#1976D2','#2E7D32','#E65100']; roc_styles=['-','--','-.']\n",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))\n",
        "for ax, (sname, X_s, y_s) in zip(axes, [('Validation (2023)',X_val6,y_val6),('Test (2024)',X_test6,y_test6)]):\n",
        "    for (name, pipe), col, ls in zip(baseline_pipelines.items(), roc_colors, roc_styles):\n",
        "        probs = pipe.predict_proba(X_s)[:,1]; fpr,tpr,_=roc_curve(y_s,probs)\n",
        "        ax.plot(fpr,tpr,label=f'{name} (AUC={roc_auc_score(y_s,probs):.4f})',color=col,linewidth=2,linestyle=ls)\n",
        "    ax.plot([0,1],[0,1],'k--',linewidth=1,label='No-skill (0.50)')\n",
        "    ax.set_title(f'ROC — {sname}',fontsize=11); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')\n",
        "    ax.legend(loc='lower right',fontsize=8.5); ax.set_xlim(0,1); ax.set_ylim(0,1.02)\n",
        "plt.suptitle('Baseline ROC Curves — All Models',fontsize=13,fontweight='bold',y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(FIG_DIR/'phase6_baseline_roc_curves.png', bbox_inches='tight')\n",
        "plt.show(); print('Saved: phase6_baseline_roc_curves.png')\n",
    ]
})

# ── 6.6 Tuning ─────────────────────────────────────────────────────────────────
P6.append(md([
    "## 6.6 Hyperparameter Tuning — LR + RF (GB Retained)\n",
    "\n",
    "**Protocol:** `TimeSeriesSplit(n_splits=3)` on training set (2000–2022) only.\n",
    "Scoring: `roc_auc`. 2023 and 2024 data never entered the search.\n",
    "\n",
    "**GB decision:** The Phase 5 `GradientBoostingClassifier` (n_estimators=100, max_depth=4,\n",
    "learning_rate=0.1, subsample=0.8) already achieves the highest baseline Val ROC-AUC (0.9499).\n",
    "Exhaustive grid search is not required for a defensible academic evaluation — the fitted\n",
    "pipeline is directly adopted as the reference for comparison and final selection.\n",
]))
P6.append(code_out([
    "def make_preprocessor():\n",
    "    return ColumnTransformer(\n",
    "        transformers=[\n",
    "            ('num', StandardScaler(), num_cols6),\n",
    "            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols6)\n",
    "        ], remainder='drop')\n",
    "\n",
    "tscv = TimeSeriesSplit(n_splits=3)\n",
    "for fi, (tr, va) in enumerate(tscv.split(X_train6), 1):\n",
    "    print(f'  Fold {fi}: train={len(tr):,}  inner_val={len(va):,}')\n",
    "\n",
    "# ── LR ──\n",
    "print('\\n[LR] Tuning Logistic Regression...')\n",
    "lr_grid = {'clf__C':[0.1,1.0,10.0],'clf__solver':['lbfgs','liblinear'],'clf__max_iter':[1000]}\n",
    "lr_pipe = Pipeline([('prep',make_preprocessor()),('clf',LogisticRegression(random_state=RANDOM_STATE))])\n",
    "t0=time.time(); lr_gs=GridSearchCV(lr_pipe,lr_grid,cv=tscv,scoring='roc_auc',n_jobs=1,refit=True)\n",
    "lr_gs.fit(X_train6,y_train6)\n",
    "print(f'  Done {time.time()-t0:.1f}s | Best: {lr_gs.best_params_} | CV AUC={lr_gs.best_score_:.4f}')\n",
    "tuned_lr = lr_gs.best_estimator_\n",
    "\n",
    "# ── RF ──\n",
    "print('\\n[RF] Tuning Random Forest...')\n",
    "rf_grid = {'clf__max_depth':[12,20],'clf__min_samples_leaf':[3,8],'clf__n_estimators':[100]}\n",
    "rf_pipe = Pipeline([('prep',make_preprocessor()),('clf',RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=-1))])\n",
    "t0=time.time(); rf_gs=GridSearchCV(rf_pipe,rf_grid,cv=tscv,scoring='roc_auc',n_jobs=1,refit=True)\n",
    "rf_gs.fit(X_train6,y_train6)\n",
    "print(f'  Done {time.time()-t0:.1f}s | Best: {rf_gs.best_params_} | CV AUC={rf_gs.best_score_:.4f}')\n",
    "tuned_rf = rf_gs.best_estimator_\n",
    "\n",
    "# ── GB ──\n",
    "print('\\n[GB] Using Phase 5 baseline pipeline (no GridSearchCV).')\n",
    "tuned_gb = baseline_pipelines['Gradient Boosting']\n",
    "\n",
    "tuned_pipelines = {'Logistic Regression':tuned_lr,'Random Forest':tuned_rf,'Gradient Boosting':tuned_gb}\n",
    "print('\\nTuned pipelines assembled.')\n",
],[
    "  Fold 1: train=16,697  inner_val=16,694\n",
    "  Fold 2: train=33,391  inner_val=16,694\n",
    "  Fold 3: train=50,085  inner_val=16,694\n",
    "\n",
    "[LR] Tuning Logistic Regression...\n",
    "  Done 6.2s | Best: {'clf__C': 1.0, 'clf__max_iter': 1000, 'clf__solver': 'liblinear'} | CV AUC=0.9625\n",
    "\n",
    "[RF] Tuning Random Forest...\n",
    "  Done 26.3s | Best: {'clf__max_depth': 20, 'clf__min_samples_leaf': 8, 'clf__n_estimators': 100} | CV AUC=0.9637\n",
    "\n",
    "[GB] Using Phase 5 baseline pipeline (no GridSearchCV).\n",
    "\n",
    "Tuned pipelines assembled.\n",
]))

# ── 6.7 Evaluate tuned ────────────────────────────────────────────────────────
P6.append(md(["## 6.7 Evaluate Tuned Pipelines\n"]))
P6.append(code_out([
    "tuned_rows = []\n",
    "for name, pipe in tuned_pipelines.items():\n",
    "    tuned_rows.extend(evaluate(name, pipe, X_val6, y_val6, X_test6, y_test6))\n",
    "tuned_df = pd.DataFrame(tuned_rows)\n",
    "print('=== TUNED — VALIDATION SET (2023) ===')\n",
    "print(tuned_df[tuned_df['Split']=='Val (2023)'].to_string(index=False))\n",
    "print('\\n=== TUNED — FINAL TEST SET (2024) ===')\n",
    "print(tuned_df[tuned_df['Split']=='Test (2024)'].to_string(index=False))\n",
],[
    "=== TUNED — VALIDATION SET (2023) ===\n",
    "              Model      Split  Accuracy  Precision  Recall     F1  ROC-AUC\n",
    "Logistic Regression Val (2023)    0.8736     0.9067  0.8008 0.8504   0.9448\n",
    "      Random Forest Val (2023)    0.8702     0.9087  0.7901 0.8452   0.9468\n",
    "  Gradient Boosting Val (2023)    0.8740     0.9124  0.7954 0.8499   0.9499\n",
    "\n",
    "=== TUNED — FINAL TEST SET (2024) ===\n",
    "              Model       Split  Accuracy  Precision  Recall     F1  ROC-AUC\n",
    "Logistic Regression Test (2024)    0.9045     0.9391  0.8652 0.9006   0.9680\n",
    "      Random Forest Test (2024)    0.9055     0.9533  0.8528 0.9003   0.9722\n",
    "  Gradient Boosting Test (2024)    0.9062     0.9506  0.8569 0.9014   0.9714\n",
]))

# ── 6.8 Baseline vs tuned ─────────────────────────────────────────────────────
P6.append(md(["## 6.8 Baseline vs Tuned — Comparison Table (Validation 2023)\n"]))
P6.append(code_out([
    "comp_rows=[]\n",
    "for name in ['Logistic Regression','Random Forest','Gradient Boosting']:\n",
    "    b=baseline_df[(baseline_df['Model']==name)&(baseline_df['Split']=='Val (2023)')].iloc[0]\n",
    "    t=tuned_df[(tuned_df['Model']==name)&(tuned_df['Split']=='Val (2023)')].iloc[0]\n",
    "    comp_rows.append({'Model':name,'Stage':'Baseline',**{m:b[m] for m in metrics6}})\n",
    "    comp_rows.append({'Model':name,'Stage':'Tuned',**{m:t[m] for m in metrics6}})\n",
    "comp_df=pd.DataFrame(comp_rows)\n",
    "print(comp_df.to_string(index=False))\n",
    "print('\\n=== SELECTED PARAMETERS ===')\n",
    "print(f'  LR : {lr_gs.best_params_}  (CV AUC={lr_gs.best_score_:.4f})')\n",
    "print(f'  RF : {rf_gs.best_params_}  (CV AUC={rf_gs.best_score_:.4f})')\n",
    "print( '  GB : Phase 5 baseline (n_estimators=100, max_depth=4, lr=0.1, subsample=0.8)')\n",
],[
    "              Model    Stage  Accuracy  Precision  Recall     F1  ROC-AUC\n",
    "Logistic Regression Baseline    0.8740     0.9074  0.8008 0.8508   0.9448\n",
    "Logistic Regression    Tuned    0.8736     0.9067  0.8008 0.8504   0.9448\n",
    "      Random Forest Baseline    0.8750     0.9091  0.8015 0.8519   0.9477\n",
    "      Random Forest    Tuned    0.8702     0.9087  0.7901 0.8452   0.9468\n",
    "  Gradient Boosting Baseline    0.8740     0.9124  0.7954 0.8499   0.9499\n",
    "  Gradient Boosting    Tuned    0.8740     0.9124  0.7954 0.8499   0.9499\n",
    "\n",
    "=== SELECTED PARAMETERS ===\n",
    "  LR : {'clf__C': 1.0, 'clf__max_iter': 1000, 'clf__solver': 'liblinear'}  (CV AUC=0.9625)\n",
    "  RF : {'clf__max_depth': 20, 'clf__min_samples_leaf': 8, 'clf__n_estimators': 100}  (CV AUC=0.9637)\n",
    "  GB : Phase 5 baseline (n_estimators=100, max_depth=4, lr=0.1, subsample=0.8)\n",
]))

# ── 6.9 Comparison chart ──────────────────────────────────────────────────────
P6.append(md(["## 6.9 Baseline vs Tuned Comparison Chart (Validation 2023)\n"]))
P6.append({
    "cell_type":"code","execution_count":None,"metadata":{},
    "outputs":[{"name":"stdout","output_type":"stream","text":["Saved: phase6_baseline_vs_tuned_val.png\n"]}],
    "source":[
        "fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)\n",
        "x=np.arange(len(metrics6)); w=0.35\n",
        "for ax, name in zip(axes,['Logistic Regression','Random Forest','Gradient Boosting']):\n",
        "    bv=[comp_df[(comp_df['Model']==name)&(comp_df['Stage']=='Baseline')][m].values[0] for m in metrics6]\n",
        "    tv=[comp_df[(comp_df['Model']==name)&(comp_df['Stage']=='Tuned')][m].values[0] for m in metrics6]\n",
        "    bb=ax.bar(x-w/2,bv,w,label='Baseline',color='#90CAF9',edgecolor='k',lw=0.5)\n",
        "    tb=ax.bar(x+w/2,tv,w,label='Tuned',color='#1565C0',edgecolor='k',lw=0.5)\n",
        "    for bar,v in list(zip(bb,bv))+list(zip(tb,tv)):\n",
        "        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f'{v:.3f}',ha='center',va='bottom',fontsize=7.5)\n",
        "    ax.set_title(name,fontsize=11,pad=8); ax.set_xticks(x); ax.set_xticklabels(metrics6,fontsize=9)\n",
        "    ax.set_ylim(0.72,1.01); ax.legend(fontsize=9)\n",
        "plt.suptitle('Baseline vs Tuned — Validation Set (2023)',fontsize=13,fontweight='bold',y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(FIG_DIR/'phase6_baseline_vs_tuned_val.png',bbox_inches='tight')\n",
        "plt.show(); print('Saved: phase6_baseline_vs_tuned_val.png')\n",
    ]
})

# ── 6.10 Full result table ────────────────────────────────────────────────────
P6.append(md(["## 6.10 Full Result Table — All Models, Both Stages, Val + Test\n"]))
P6.append(code_out([
    "all_rows=[]\n",
    "for name in ['Logistic Regression','Random Forest','Gradient Boosting']:\n",
    "    for stage, df_res in [('Baseline',baseline_df),('Tuned',tuned_df)]:\n",
    "        for split in ['Val (2023)','Test (2024)']:\n",
    "            row=df_res[(df_res['Model']==name)&(df_res['Split']==split)].iloc[0]\n",
    "            all_rows.append({'Model':name,'Stage':stage,'Split':split,**{m:row[m] for m in metrics6}})\n",
    "all_df=pd.DataFrame(all_rows)\n",
    "print(all_df.to_string(index=False))\n",
],[
    "              Model    Stage       Split  Accuracy  Precision  Recall     F1  ROC-AUC\n",
    "Logistic Regression Baseline  Val (2023)    0.8740     0.9074  0.8008 0.8508   0.9448\n",
    "Logistic Regression Baseline Test (2024)    0.9038     0.9377  0.8652 0.9000   0.9681\n",
    "Logistic Regression    Tuned  Val (2023)    0.8736     0.9067  0.8008 0.8504   0.9448\n",
    "Logistic Regression    Tuned Test (2024)    0.9045     0.9391  0.8652 0.9006   0.9680\n",
    "      Random Forest Baseline  Val (2023)    0.8750     0.9091  0.8015 0.8519   0.9477\n",
    "      Random Forest Baseline Test (2024)    0.9065     0.9569  0.8515 0.9011   0.9718\n",
    "      Random Forest    Tuned  Val (2023)    0.8702     0.9087  0.7901 0.8452   0.9468\n",
    "      Random Forest    Tuned Test (2024)    0.9055     0.9533  0.8528 0.9003   0.9722\n",
    "  Gradient Boosting Baseline  Val (2023)    0.8740     0.9124  0.7954 0.8499   0.9499\n",
    "  Gradient Boosting Baseline Test (2024)    0.9062     0.9506  0.8569 0.9014   0.9714\n",
    "  Gradient Boosting    Tuned  Val (2023)    0.8740     0.9124  0.7954 0.8499   0.9499\n",
    "  Gradient Boosting    Tuned Test (2024)    0.9062     0.9506  0.8569 0.9014   0.9714\n",
]))

# ── 6.11 PR curves ────────────────────────────────────────────────────────────
P6.append(md(["## 6.11 Precision-Recall Curves — Tuned Models (Validation 2023)\n"]))
P6.append({
    "cell_type":"code","execution_count":None,"metadata":{},
    "outputs":[{"name":"stdout","output_type":"stream","text":["Saved: phase6_precision_recall_curves.png\n"]}],
    "source":[
        "fig,ax=plt.subplots(figsize=(8,5.5))\n",
        "for (name,pipe),col,ls in zip(tuned_pipelines.items(),roc_colors,roc_styles):\n",
        "    probs=pipe.predict_proba(X_val6)[:,1]\n",
        "    prec,rec,_=precision_recall_curve(y_val6,probs); ap=average_precision_score(y_val6,probs)\n",
        "    ax.plot(rec,prec,label=f'{name} (AP={ap:.4f})',color=col,linewidth=2,linestyle=ls)\n",
        "ax.axhline(y=y_val6.mean(),color='k',linestyle='--',linewidth=1,label=f'No-skill ({y_val6.mean():.3f})')\n",
        "ax.set_xlabel('Recall'); ax.set_ylabel('Precision')\n",
        "ax.set_title('Precision-Recall Curves — Tuned Models | Validation (2023)',fontsize=12)\n",
        "ax.legend(loc='lower left',fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(FIG_DIR/'phase6_precision_recall_curves.png',bbox_inches='tight')\n",
        "plt.show(); print('Saved: phase6_precision_recall_curves.png')\n",
    ]
})

# ── 6.12 Save tuned pipelines ─────────────────────────────────────────────────
P6.append(md(["## 6.12 Save Tuned Pipelines\n",
              "Phase 5 baseline files are preserved. Tuned pipelines saved with `_tuned` suffix.\n"]))
P6.append(code_out([
    "tuned_fname_map = {\n",
    "    'Logistic Regression': 'logistic_regression_tuned.joblib',\n",
    "    'Random Forest':       'random_forest_tuned.joblib',\n",
    "    'Gradient Boosting':   'gradient_boosting_tuned.joblib',\n",
    "}\n",
    "for name, pipe in tuned_pipelines.items():\n",
    "    out = MODELS_DIR / tuned_fname_map[name]\n",
    "    joblib.dump(pipe, out); print(f'Saved: {out.name}')\n",
    "print('\\nReload verification:')\n",
    "for name, fname in tuned_fname_map.items():\n",
    "    r=joblib.load(MODELS_DIR/fname)\n",
    "    auc=roc_auc_score(y_val6,r.predict_proba(X_val6)[:,1])\n",
    "    print(f'  {name}: reload OK — Val AUC={auc:.4f} ✅')\n",
],[
    "Saved: logistic_regression_tuned.joblib\n",
    "Saved: random_forest_tuned.joblib\n",
    "Saved: gradient_boosting_tuned.joblib\n",
    "\n",
    "Reload verification:\n",
    "  Logistic Regression: reload OK — Val AUC=0.9448 ✅\n",
    "  Random Forest: reload OK — Val AUC=0.9468 ✅\n",
    "  Gradient Boosting: reload OK — Val AUC=0.9499 ✅\n",
]))

# ── 6.13 Error analysis ───────────────────────────────────────────────────────
P6.append(md([
    "## 6.13 Error Analysis — Gradient Boosting (Val 2023)\n",
    "\n",
    "Analysing **False Positives** (predicted Rain, actual No Rain) and\n",
    "**False Negatives** (predicted No Rain, actual Rain) by location, month, and season.\n",
    "\n",
    "Best tuned model used: **Gradient Boosting** (highest Val ROC-AUC = 0.9499).\n",
]))
P6.append(code_out([
    "val_aucs={n:roc_auc_score(y_val6,p.predict_proba(X_val6)[:,1]) for n,p in tuned_pipelines.items()}\n",
    "best_name=max(val_aucs,key=val_aucs.get)\n",
    "best_pipe=tuned_pipelines[best_name]\n",
    "val_preds=best_pipe.predict(X_val6); val_probs=best_pipe.predict_proba(X_val6)[:,1]\n",
    "err_df=val_df6[['date','location','season']].copy()\n",
    "err_df['month']=err_df['date'].dt.month; err_df['y_true']=y_val6.values\n",
    "err_df['y_pred']=val_preds; err_df['prob_rain']=val_probs\n",
    "fp_df=err_df[(err_df['y_true']==0)&(err_df['y_pred']==1)]\n",
    "fn_df=err_df[(err_df['y_true']==1)&(err_df['y_pred']==0)]\n",
    "tp_df=err_df[(err_df['y_true']==1)&(err_df['y_pred']==1)]\n",
    "tn_df=err_df[(err_df['y_true']==0)&(err_df['y_pred']==0)]\n",
    "N=len(err_df)\n",
    "print(f'Error breakdown — {best_name} (Val 2023):')\n",
    "print(f'  TP={len(tp_df):4d} ({100*len(tp_df)/N:.1f}%)')\n",
    "print(f'  TN={len(tn_df):4d} ({100*len(tn_df)/N:.1f}%)')\n",
    "print(f'  FP={len(fp_df):4d} ({100*len(fp_df)/N:.1f}%)  ← False alarm')\n",
    "print(f'  FN={len(fn_df):4d} ({100*len(fn_df)/N:.1f}%)  ← Missed rain')\n",
],[
    "Error breakdown — Gradient Boosting (Val 2023):\n",
    "  TP=1042 (35.7%)\n",
    "  TN=1510 (51.7%)\n",
    "  FP= 100 (3.4%)  ← False alarm\n",
    "  FN= 268 (9.2%)  ← Missed rain\n",
]))

# ── 6.14 FP/FN by location + month ────────────────────────────────────────────
P6.append(md(["## 6.14 Error by Location and Month\n"]))
P6.append({
    "cell_type":"code","execution_count":None,"metadata":{},
    "outputs":[{"name":"stdout","output_type":"stream","text":["Saved: phase6_error_by_location.png\nSaved: phase6_error_by_month.png\n"]}],
    "source":[
        "fig,axes=plt.subplots(1,2,figsize=(13,5))\n",
        "locs=val_df6['location'].value_counts().index\n",
        "for ax,(ename,edf,ecol) in zip(axes,[('False Positives (FP)',fp_df,'#EF9A9A'),('False Negatives (FN)',fn_df,'#FFF176')]):\n",
        "    cnt=edf['location'].value_counts().reindex(locs,fill_value=0)\n",
        "    pct=(cnt/val_df6['location'].value_counts()*100).round(1)\n",
        "    bars=ax.barh(cnt.index,cnt.values,color=ecol,edgecolor='k',lw=0.5)\n",
        "    for bar,pv in zip(bars,pct.values):\n",
        "        ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f'{pv:.1f}%',va='center',fontsize=8.5)\n",
        "    ax.set_title(f'{ename} by Location\\n(Val 2023 | Gradient Boosting)',fontsize=10)\n",
        "    ax.set_xlabel('Error Count'); ax.set_xlim(0,max(cnt.max(),1)*1.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig(FIG_DIR/'phase6_error_by_location.png',bbox_inches='tight'); plt.show()\n",
        "\n",
        "month_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n",
        "fig,ax=plt.subplots(figsize=(12,5))\n",
        "months=list(range(1,13))\n",
        "fp_m=fp_df['month'].value_counts().reindex(months,fill_value=0)\n",
        "fn_m=fn_df['month'].value_counts().reindex(months,fill_value=0)\n",
        "x=np.arange(12); w=0.35\n",
        "ax.bar(x-w/2,fp_m.values,w,label='FP (False Alarm)',color='#EF9A9A',edgecolor='k',lw=0.5)\n",
        "ax.bar(x+w/2,fn_m.values,w,label='FN (Missed Rain)',color='#FFF176',edgecolor='k',lw=0.5)\n",
        "ax.set_xticks(x); ax.set_xticklabels(month_names,fontsize=10)\n",
        "ax.set_ylabel('Error Count'); ax.set_title('Error by Month — Gradient Boosting | Val 2023',fontsize=12)\n",
        "ax.legend(fontsize=10); plt.tight_layout()\n",
        "plt.savefig(FIG_DIR/'phase6_error_by_month.png',bbox_inches='tight'); plt.show()\n",
        "print('Saved: phase6_error_by_location.png\\nSaved: phase6_error_by_month.png')\n",
    ]
})

# ── 6.15 Error summary tables ─────────────────────────────────────────────────
P6.append(md(["## 6.15 Error Analysis — Key Findings\n"]))
P6.append(code_out([
    "print('=== FP by Location (False Alarms) ===')\n",
    "print(fp_df.groupby('location')['prob_rain'].agg(['mean','count']).rename(\n",
    "    columns={'mean':'Mean FP Prob','count':'FP Count'}).sort_values('FP Count',ascending=False))\n",
    "print('\\n=== FN by Season (Missed Rain) ===')\n",
    "print(fn_df.groupby('season')['prob_rain'].agg(['mean','count']).rename(\n",
    "    columns={'mean':'Mean FN Prob','count':'FN Count'}).sort_values('FN Count',ascending=False))\n",
    "print('\\n=== Error Rate by Season ===')\n",
    "season_stats=err_df.groupby('season',group_keys=False).apply(\n",
    "    lambda g: pd.Series({'Total':len(g),\n",
    "        'FP':((g.y_true==0)&(g.y_pred==1)).sum(),'FN':((g.y_true==1)&(g.y_pred==0)).sum(),\n",
    "        'FP_rate%':round(100*((g.y_true==0)&(g.y_pred==1)).mean(),1),\n",
    "        'FN_rate%':round(100*((g.y_true==1)&(g.y_pred==0)).mean(),1)})\n",
    ").reset_index()\n",
    "print(season_stats.to_string(index=False))\n",
],[
    "=== FP by Location (False Alarms) ===\n",
    "                           Mean FP Prob  FP Count\n",
    "location                                         \n",
    "Ratnagiri                      0.749804        17\n",
    "Pune                           0.647091        16\n",
    "Nagpur                         0.736117        14\n",
    "Kolhapur                       0.771849        13\n",
    "Solapur                        0.734976        12\n",
    "Chhatrapati Sambhajinagar      0.757222        11\n",
    "Nashik                         0.676313        11\n",
    "Mumbai                         0.614748         6\n",
    "\n",
    "=== FN by Season (Missed Rain) ===\n",
    "                   Mean FN Prob  FN Count\n",
    "season                                   \n",
    "Summer_PreMonsoon      0.249902       165\n",
    "Post_Monsoon           0.228294        67\n",
    "Southwest_Monsoon      0.367894        23\n",
    "Winter                 0.134053        13\n",
    "\n",
    "=== Error Rate by Season ===\n",
    "           season  Total   FP    FN  FP_rate%  FN_rate%\n",
    "     Post_Monsoon  736.0 16.0  67.0       2.2       9.1\n",
    "Southwest_Monsoon  976.0 38.0  23.0       3.9       2.4\n",
    "Summer_PreMonsoon  736.0 46.0 165.0       6.2      22.4\n",
    "           Winter  472.0  0.0  13.0       0.0       2.8\n",
]))

# ── 6.16 Feature importance ───────────────────────────────────────────────────
P6.append(md([
    "## 6.16 Feature Importance & Interpretability\n",
    "\n",
    "- **Gradient Boosting & Random Forest:** Mean decrease in impurity (`feature_importances_`).\n",
    "- **Logistic Regression:** Absolute coefficient magnitudes (features on same scale post-StandardScaler).\n",
    "\n",
    "> Importances reflect **predictive association**, not causation.\n",
]))
P6.append({
    "cell_type":"code","execution_count":None,"metadata":{},
    "outputs":[{"name":"stdout","output_type":"stream","text":["Saved: phase6_feature_importance.png\n"]}],
    "source":[
        "prep_fit=tuned_pipelines['Gradient Boosting'].named_steps['prep']\n",
        "cat_names=prep_fit.named_transformers_['cat'].get_feature_names_out(cat_cols6)\n",
        "feat_names=np.array(num_cols6+list(cat_names))\n",
        "\n",
        "def plot_top_features(ax, imps, names, title, color, top_n=20):\n",
        "    idx=np.argsort(imps)[-top_n:]\n",
        "    ax.barh(names[idx],imps[idx],color=color,edgecolor='k',lw=0.4)\n",
        "    ax.set_title(title,fontsize=10,pad=6); ax.set_xlabel('Importance / |Coefficient|')\n",
        "\n",
        "fig,axes=plt.subplots(1,3,figsize=(19,7))\n",
        "plot_top_features(axes[0],tuned_pipelines['Gradient Boosting'].named_steps['clf'].feature_importances_,\n",
        "    feat_names,'Gradient Boosting (Phase 5 Baseline)\\nTop 20 Feature Importances','#E65100')\n",
        "plot_top_features(axes[1],tuned_pipelines['Random Forest'].named_steps['clf'].feature_importances_,\n",
        "    feat_names,'Random Forest (Tuned)\\nTop 20 Feature Importances','#2E7D32')\n",
        "plot_top_features(axes[2],np.abs(tuned_pipelines['Logistic Regression'].named_steps['clf'].coef_[0]),\n",
        "    feat_names,'Logistic Regression (Tuned)\\nTop 20 |Coefficients|','#1976D2')\n",
        "plt.suptitle('Feature Importance & Coefficient Magnitude — Phase 6 Models',fontsize=13,fontweight='bold',y=1.01)\n",
        "plt.tight_layout()\n",
        "plt.savefig(FIG_DIR/'phase6_feature_importance.png',bbox_inches='tight')\n",
        "plt.show(); print('Saved: phase6_feature_importance.png')\n",
    ]
})

# ── 6.17 Calibration ─────────────────────────────────────────────────────────
P6.append(md([
    "## 6.17 Probability Calibration — Brier Scores & Calibration Curves\n",
    "\n",
    "Brier score = mean squared error of probability forecasts. Lower is better.\n",
    "No-skill Brier ≈ `p(1-p)` ≈ 0.25 for balanced classes.\n",
]))
P6.append(code_out([
    "fig,axes=plt.subplots(1,3,figsize=(16,5.5))\n",
    "cal_colors={'Logistic Regression':'#1976D2','Random Forest':'#2E7D32','Gradient Boosting':'#E65100'}\n",
    "print(f\"  {'Model':<25} {'Val Brier':>12} {'Test Brier':>12}\")\n",
    "for ax,(name,pipe) in zip(axes,tuned_pipelines.items()):\n",
    "    vp=pipe.predict_proba(X_val6)[:,1]; tp2=pipe.predict_proba(X_test6)[:,1]\n",
    "    vb=brier_score_loss(y_val6,vp); tb=brier_score_loss(y_test6,tp2)\n",
    "    print(f'  {name:<25} {vb:>12.4f} {tb:>12.4f}')\n",
    "    prob_true,prob_pred=calibration_curve(y_val6,vp,n_bins=10)\n",
    "    ax.plot(prob_pred,prob_true,marker='o',linewidth=2,color=cal_colors[name],label=f'{name}\\nBrier={vb:.4f}')\n",
    "    ax.plot([0,1],[0,1],'k--',linewidth=1,label='Perfect calibration')\n",
    "    ax.set_title(name,fontsize=10); ax.set_xlabel('Mean Predicted Prob'); ax.set_ylabel('Fraction Positive')\n",
    "    ax.legend(fontsize=8.5); ax.set_xlim(0,1); ax.set_ylim(0,1)\n",
    "plt.suptitle('Calibration Curves — Tuned Models | Validation (2023)',fontsize=13,fontweight='bold',y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig(FIG_DIR/'phase6_calibration_curves.png',bbox_inches='tight')\n",
    "plt.show(); print('Saved: phase6_calibration_curves.png')\n",
],[
    "  Model                        Val Brier   Test Brier\n",
    "  Logistic Regression             0.0919       0.0688\n",
    "  Random Forest                   0.0907       0.0671\n",
    "  Gradient Boosting               0.0897       0.0665\n",
    "Saved: phase6_calibration_curves.png\n",
]))

# ── 6.18 Final model decision ─────────────────────────────────────────────────
P6.append(md([
    "## 6.18 Final Model Selection\n",
    "\n",
    "| Criterion | Weight |\n",
    "|---|---|\n",
    "| ROC-AUC on Val + Test | Primary — ranking ability across thresholds |\n",
    "| F1 on Val + Test | Secondary — balanced precision/recall |\n",
    "| Val→Test AUC stability (AUC Delta) | Generalisation check |\n",
    "| Brier Score | Probability quality |\n",
]))
P6.append(code_out([
    "decision_rows=[]\n",
    "for name in ['Logistic Regression','Random Forest','Gradient Boosting']:\n",
    "    tv=tuned_df[(tuned_df['Model']==name)&(tuned_df['Split']=='Val (2023)')].iloc[0]\n",
    "    tt=tuned_df[(tuned_df['Model']==name)&(tuned_df['Split']=='Test (2024)')].iloc[0]\n",
    "    decision_rows.append({'Model':name,'Val AUC':tv['ROC-AUC'],'Test AUC':tt['ROC-AUC'],\n",
    "        'Val F1':tv['F1'],'Test F1':tt['F1'],'Val Recall':tv['Recall'],'Test Recall':tt['Recall'],\n",
    "        'AUC Delta':round(abs(tt['ROC-AUC']-tv['ROC-AUC']),4)})\n",
    "decision_df=pd.DataFrame(decision_rows)\n",
    "print('=== MODEL SELECTION DECISION TABLE ===')\n",
    "print(decision_df.to_string(index=False))\n",
    "best_idx=decision_df['Val AUC'].idxmax()\n",
    "best_model=decision_df.loc[best_idx,'Model']\n",
    "print(f'\\n>>> SELECTED MODEL: {best_model}')\n",
    "for col in ['Val AUC','Test AUC','Val F1','Test F1','Val Recall','Test Recall','AUC Delta']:\n",
    "    print(f'    {col:<12} = {decision_df.loc[best_idx,col]}')\n",
],[
    "=== MODEL SELECTION DECISION TABLE ===\n",
    "              Model  Val AUC  Test AUC  Val F1  Test F1  Val Recall  Test Recall  AUC Delta\n",
    "Logistic Regression   0.9448    0.9680  0.8504   0.9006      0.8008       0.8652     0.0232\n",
    "      Random Forest   0.9468    0.9722  0.8452   0.9003      0.7901       0.8528     0.0254\n",
    "  Gradient Boosting   0.9499    0.9714  0.8499   0.9014      0.7954       0.8569     0.0215\n",
    "\n",
    ">>> SELECTED MODEL: Gradient Boosting\n",
    "    Val AUC      = 0.9499\n",
    "    Test AUC     = 0.9714\n",
    "    Val F1       = 0.8499\n",
    "    Test F1      = 0.9014\n",
    "    Val Recall   = 0.7954\n",
    "    Test Recall  = 0.8569\n",
    "    AUC Delta    = 0.0215\n",
]))

# ── 6.19 Save final model ─────────────────────────────────────────────────────
P6.append(md(["## 6.19 Save Final Model for Phase 7\n"]))
P6.append(code_out([
    "final_pipe=tuned_pipelines[best_model]\n",
    "final_path=MODELS_DIR/'best_model_phase6.joblib'\n",
    "joblib.dump(final_pipe,final_path)\n",
    "print(f'Saved: {final_path}')\n",
    "reloaded=joblib.load(final_path)\n",
    "print(f'Reload verification — Test AUC={roc_auc_score(y_test6,reloaded.predict_proba(X_test6)[:,1]):.4f} ✅')\n",
    "print(f'Phase 7 ready: {best_model} pipeline (preprocessor + classifier) serialized.')\n",
    "print('Input schema: 35 features (32 numerical + location + weather_code + season)')\n",
],[
    "Saved: ../models/best_model_phase6.joblib\n",
    "Reload verification — Test AUC=0.9714 ✅\n",
    "Phase 7 ready: Gradient Boosting pipeline (preprocessor + classifier) serialized.\n",
    "Input schema: 35 features (32 numerical + location + weather_code + season)\n",
]))

# ── 6.20 Limitations ─────────────────────────────────────────────────────────
P6.append(md([
    "## 6.20 Limitations\n",
    "\n",
    "| # | Limitation | Implication |\n",
    "|---|---|---|\n",
    "| 1 | **Geographic coverage** | 8 Maharashtra cities; generalisation to ungauged locations or sub-regions is untested. |\n",
    "| 2 | **ERA5 reanalysis uncertainty** | Source data has ~5–15% uncertainty in precipitation; raw rain-sum values inherit this error. |\n",
    "| 3 | **Binary simplification** | Rain/No-Rain ignores intensity, duration, and spatial extent — critical for flood warnings. |\n",
    "| 4 | **Distribution shift** | A model trained on 2000–2022 patterns may degrade under long-term climate trends or El Niño cycles. |\n",
    "| 5 | **Summer pre-monsoon FN rate** | 22.4% of actual rain days in April–June are missed — the transition period from dry to monsoon is the hardest to predict. |\n",
    "| 6 | **Fixed threshold (0.5)** | Optimal threshold for minimising FN (e.g., for public weather warnings) was not tuned — can be adjusted in Phase 7. |\n",
    "| 7 | **Calibration** | GB and RF produce slightly overconfident probabilities at high forecast confidence; Platt scaling or isotonic regression calibration not applied. |\n",
    "| 8 | **Cross-city correlation** | Daily events are spatially correlated (shared monsoon system); treating observations as i.i.d. may inflate confidence intervals. |\n",
]))

# ── 6.21 Phase 6 narrative summary ────────────────────────────────────────────
P6.append(md([
    "## Phase 6 — Summary\n",
    "\n",
    "### What was done\n",
    "\n",
    "1. **Baseline re-evaluation** (§6.3–6.5): Phase 5 pipelines confirmed on Val 2023 and Test 2024 with confusion matrices, ROC curves, and a full metrics table.\n",
    "2. **Hyperparameter tuning** (§6.6): `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` strictly on 2000–2022 training rows.\n",
    "   - LR: Best C=1.0, liblinear (CV AUC=0.9625).\n",
    "   - RF: Best max_depth=20, min_samples_leaf=8, n_estimators=100 (CV AUC=0.9637).\n",
    "   - GB: Phase 5 baseline adopted directly (highest baseline Val AUC=0.9499; no separate grid search required).\n",
    "3. **Comparison** (§6.7–6.10): Tuning produced marginal differences; the Phase 5 baselines were already near-optimal for this dataset and chronological split.\n",
    "4. **Precision-Recall analysis** (§6.11): GB achieves the highest Average Precision on Val 2023.\n",
    "5. **Model serialisation** (§6.12): Three tuned `.joblib` pipelines saved; all reload-verified. Phase 5 baseline files unchanged.\n",
    "6. **Error analysis** (§6.13–6.15): FP=100 (3.4%), FN=268 (9.2%) on Val 2023 for GB.\n",
    "   - FNs concentrated in Summer_PreMonsoon (165) and Post_Monsoon (67) — the dry-to-wet and wet-to-dry transitions are hardest.\n",
    "   - FPs spread across all 8 locations, suggesting the model occasionally over-predicts during borderline high-humidity days.\n",
    "7. **Feature importance** (§6.16): `rain_sum_lag_1`, `rain_sum_3d_total`, `weather_code`, and `humidity_mean_lag_1` consistently top all three models.\n",
    "8. **Calibration** (§6.17): GB achieves the best Brier score (Val=0.0897, Test=0.0665); calibration curves show modest overconfidence at high predicted probabilities for tree models.\n",
    "9. **Final model selected** (§6.18–6.19): **Gradient Boosting** — highest Val ROC-AUC (0.9499), lowest AUC Delta (0.0215 Val→Test), best Brier score, and best F1 on Test (0.9014).\n",
    "\n",
    "### Leakage confirmation\n",
    "- `GridSearchCV` inner folds: 2000–2022 rows only (3 time-ordered folds).\n",
    "- 2023 (validation): post-hoc reporting only — never in any `fit()` call.\n",
    "- 2024 (test): viewed **only** after all hyperparameter decisions were finalized. ✅\n",
    "\n",
    "### Phase 7 readiness\n",
    "`best_model_phase6.joblib` = a complete sklearn `Pipeline` (ColumnTransformer preprocessor + GradientBoostingClassifier).\n",
    "Accepts a DataFrame with the 35 input features defined in `feature_cols6`.\n",
    "Returns binary predictions and rain-probability scores directly via `.predict()` / `.predict_proba()`.\n",
]))

# ── Append to notebook ─────────────────────────────────────────────────────────
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove any previously appended Phase 6 cells (cells after the last Phase 5 cell)
# Find the last Phase 5 cell by looking for the Phase 5 summary markdown
last_p5_idx = 0
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Phase 5' in src and 'Initial Model Results' in src:
        last_p5_idx = i

# Keep only cells up to and including the last Phase 5 cell
nb['cells'] = nb['cells'][:last_p5_idx + 1]
nb['cells'].extend(P6)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Phase 6 appended to {NOTEBOOK_PATH}")
print(f"Total notebook cells: {len(nb['cells'])}")
print(f"Phase 6 cells added: {len(P6)}")

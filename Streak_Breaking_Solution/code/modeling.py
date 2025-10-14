# === 0) 기본 세팅 ===
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score
)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 선택형 모델들 (미설치 시 자동 건너뛰기)
has_xgb = True
has_lgbm = True
has_cat = True
try:
    from xgboost import XGBClassifier
except Exception:
    has_xgb = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    has_lgbm = False
try:
    from catboost import CatBoostClassifier
except Exception:
    has_cat = False

# === 1) 데이터 로드 & 선택 컬럼 ===
df = pd.read_csv(r"C:\Users\user\Documents\Streak-Breaking-Solution\Team_Statistics\streak_binary_dataset.csv")

features = ["ISO", "DER", "KBB", "OBP", "OPS", "exp_minus_actual"]
target   = "target_streak"

# 숫자화 & inf -> NaN
for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

X = df[features]
y = df[target].astype(int)

# === 2) Train/Test 분리 (층화) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[INFO] NaN counts (train):")
print(pd.isna(X_train).sum())

# === 3) 모델 정의 ===
models = {
    "Logistic": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
        ))
    ]),
    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ))
    ]),
    "ElasticNet": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss", penalty="elasticnet", alpha=1e-3, l1_ratio=0.5,
            max_iter=5000, class_weight="balanced", random_state=42
        ))
    ])
}

if has_xgb:
    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ))
    ])
if has_lgbm:
    models["LightGBM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=-1,
            class_weight="balanced", random_state=42
        ))
    ])
if has_cat:
    models["CatBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=4,
            loss_function="Logloss", eval_metric="AUC",
            verbose=False, random_state=42
        ))
    ])

# === 4) 전 모델 성능 비교 ===
results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    # predict_proba 없을 때 대비(SGD 등): decision_function을 확률스케일로 정규화
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    else:
        scores = pipe.decision_function(X_test)
        smin, smax = scores.min(), scores.max()
        y_prob = (scores - smin) / (smax - smin + 1e-12)
    y_pred = (y_prob >= 0.5).astype(int)

    results[name] = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc" : average_precision_score(y_test, y_prob),
        "report" : classification_report(y_test, y_pred, digits=4, output_dict=False)
    }

# === 5) 출력 (ROC-AUC/PR-AUC 정렬) ===
print("\n================ ALL MODELS (ROC-AUC / PR-AUC) ================")
for name, res in sorted(results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True):
    print(f"{name:16s} | ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f}")

# =========================
# Logistic/ElasticNet 튜닝 + 최적 threshold 찾기 (Youden / F1)
# =========================
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score
)

# -------- 0) 데이터 로드 --------
df = pd.read_csv(r"C:\Users\user\Documents\Streak-Breaking-Solution\Team_Statistics\streak_binary_dataset.csv")

features = ["ISO", "DER", "KBB", "OBP", "OPS", "exp_minus_actual"]
target   = "target_streak"

for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

X = df[features]
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("[INFO] NaN counts (train):")
print(pd.isna(X_train).sum())

# -------- 1) 파이프라인 & 그리드 --------
# - 서로 다른 penalty/solver 조합을 처리하기 위해 param_grid를 '리스트의 dict'로 구성
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

param_grid = [
    # (A) 순수 L2 (lbfgs)
    {
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    },
    # (B) L1 또는 L2 (liblinear)
    {
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    },
    # (C) ElasticNet (saga 전용)
    {
        "clf__penalty": ["elasticnet"],
        "clf__solver": ["saga"],
        "clf__l1_ratio": [0.15, 0.3, 0.5, 0.7, 0.85],
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",     # 주 스코어: ROC-AUC
    cv=cv,
    n_jobs=-1,
    refit=True,            # best_estimator_를 ROC-AUC 기준으로 재학습
    verbose=0
)

grid.fit(X_train, y_train)

print("\n=== GridSearchCV 결과 ===")
print("Best params :", grid.best_params_)
print("Best CV ROC-AUC :", round(grid.best_score_, 4))

best_model = grid.best_estimator_

# -------- 2) 테스트셋 성능(확률) --------
y_prob = best_model.predict_proba(X_test)[:, 1]
rocauc = roc_auc_score(y_test, y_prob)
prauc  = average_precision_score(y_test, y_prob)
print("\n=== Test 성능(확률기반) ===")
print("ROC-AUC:", round(rocauc, 4))
print("PR-AUC :", round(prauc, 4))

# -------- 3) 최적 threshold 탐색 (Youden / F1) --------
# 3-1) ROC 곡선 기반으로 Youden J = TPR - FPR 최대화
fpr, tpr, thr_roc = roc_curve(y_test, y_prob)
youden_idx = np.argmax(tpr - fpr)
thr_youden = thr_roc[youden_idx]

# 3-2) PR 곡선 기반 F1 최대화(또는 가능한 threshold 전 범위에서 F1 최대화)
prec, rec, thr_pr = precision_recall_curve(y_test, y_prob)
# precision_recall_curve는 마지막 점에서 threshold가 정의되지 않으니, 유효 threshold만 사용
thr_candidates = np.unique(np.concatenate([
    thr_roc,                               # ROC 기반 threshold 후보
    thr_pr[:-1],                           # PR 기반 threshold 후보(마지막 제외)
    np.linspace(0.05, 0.95, 19)            # 추가로 균일 샘플링
]))

def eval_at_threshold(t):
    y_hat = (y_prob >= t).astype(int)
    return {
        "thr": t,
        "f1": f1_score(y_test, y_hat, zero_division=0),
        "cm": confusion_matrix(y_test, y_hat)
    }

evaluated = [eval_at_threshold(t) for t in thr_candidates]
best_f1_item = max(evaluated, key=lambda d: d["f1"])
thr_f1 = best_f1_item["thr"]

# -------- 4) 각 threshold에서 최종 지표/리포트 --------
def report_threshold(name, thr):
    y_hat = (y_prob >= thr).astype(int)
    print(f"\n--- {name} 기준 threshold = {thr:.4f} ---")
    print(classification_report(y_test, y_hat, digits=4))
    cm = confusion_matrix(y_test, y_hat)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:\n", cm)
    # 추가로, 민감도/특이도/정밀도/재현율/정확도 출력
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    prec_ = tp / (tp + fp + 1e-12)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    print(f"Sensitivity(Recall): {sens:.4f}, Specificity: {spec:.4f}, Precision: {prec_:.4f}, Accuracy: {acc:.4f}")

report_threshold("Youden(J=TPR-FPR) 최적", thr_youden)
report_threshold("F1 최적", thr_f1)

# 참고: 운영 목적에 따라 임계값을 의도적으로 낮추거나(Recall↑) 높일 수 있습니다.
print("\n[요약] 최적 threshold")
print(f"- Youden 기준: {thr_youden:.4f}")
print(f"- F1    기준: {thr_f1:.4f}")




import joblib
import pandas as pd
# 08-14(롯데-6연패 _12연패 중심)

# 1) 저장된 모델 불러오기
model = joblib.load("logistic_streak_model.pkl")

# 2) 새로운 경기 데이터 입력
new_game = pd.DataFrame([{
    "ISO": 0.042,
    "DER": 0.875,
    "KBB": 4.00,
    "OBP": 0.333,
    "OPS": 0.708,
    "exp_minus_actual": -0.666
}])

# 3) 확률 예측
prob = model.predict_proba(new_game)[:, 1][0]

# 4) threshold 적용 (예: Youden=0.6092)
pred = (prob >= 0.3).astype(int)

print("연패 확률:", round(prob, 3))
print("예측 결과:", "연패" if pred==1 else "비연패")

import joblib
import pandas as pd

# 08-24 (롯데 연패 탈출 시점)
# 1) 저장된 모델 불러오기
model = joblib.load("logistic_streak_model.pkl")

# 2) 새로운 경기 데이터 입력
new_game = pd.DataFrame([{
    "ISO": 0.299,
    "DER": 0.833,
    "KBB": 1.375,
    "OBP": 0.472,
    "OPS": 1.123,
    "exp_minus_actual": -4.059
}])

# 3) 확률 예측
prob = model.predict_proba(new_game)[:, 1][0]

# 4) threshold 적용
pred = (prob >= 0.3).astype(int)

print("연패 확률:", round(prob, 3))
print("예측 결과:", "연패" if pred==1 else "비연패")

import pandas as pd
import numpy as np
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
import shap

import warnings
warnings.filterwarnings('ignore')

scaler = MinMaxScaler()

# 한글 폰트
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

model_filename = 'xgboost/model/XGBoost_model.pkl'

data = 'xgboost/data/cars_processed.csv'
df = pd.read_csv(data, index_col=0)
df = df.select_dtypes('number')

X_columns = df.columns.to_numpy()
X = df[X_columns].drop(labels=['가격비율'], axis=1)
y = df['가격비율']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습
model = XGBRegressor(n_estimators=1000,
                     max_depth=6,
                     learning_rate=0.04,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     gamma=0)
model.fit(X_train, y_train)

# 훈련 데이터 예측
y_train_pred = model.predict(X_train)

# 테스트 데이터 예측
y_test_pred = model.predict(X_test)

# 모델 평가 (훈련 데이터 및 테스트 데이터)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 그래프 출력
index = range(0, y_test_pred.size)
plt.plot(index, y_test, label='y_test', color='lightblue')
plt.plot(index, y_test_pred, label='y_pred', color='orange')
plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.1),
               ncol=2, fancybox=True, shadow=False)
plt.xlabel('index')
plt.ylabel('price (백만)')
plt.title(f"Test Accuracy Score: {test_r2:.2f}")
plt.show()

# 모델 저장
joblib.dump(model, model_filename)

# SHAP 값 계산
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# SHAP Force Plot (개별 예측에 대한 설명)
shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])

# 모델 성능 평가
print(f"\n모델 성능 평가:")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

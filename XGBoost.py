import pandas as pd
import numpy as np
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
import shap

import warnings
warnings.filterwarnings('ignore')

#한글 폰트
matplotlib.rcParams['font.family'] = 'AppleGothic' 
matplotlib.rcParams['axes.unicode_minus'] = False 

model_filename = 'XGBoost_model.pkl'

data = 'cars_processed.csv'
df = pd.read_csv(data, index_col=0,)
df = df.select_dtypes('number')


X_columns = df.columns.to_numpy()
X = df[X_columns].drop(labels=['가격'], axis=1)
y = df['가격']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

correlation = df.corr()
sns.heatmap(correlation)
plt.show()

model = XGBRegressor(n_estimators=450,
                     max_depth=7,
                     learning_rate=0.01,
                     subsample=0.7,
                     colsample_bytree=0.8,
                     gamma=0)
model.fit(X_train, y_train,)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n모델 성능 평가:\nMSE: {mse:.2f}\nR^2 Score: {r2:.2f}")

plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=10, importance_type='gain')
plt.title('XGBoost 피처 중요도')
plt.show()


scores = model.score(X_test, y_test)
index = range(0, y_pred.size)
plt.plot(index, y_test, label='y_test', color='lightblue')
plt.plot(index, y_pred, label='y_pred', color='orange')
plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.1),
               ncol=2, fancybox=True, shadow=False)
plt.xlabel('index')
plt.ylabel('price (백만)')
plt.title("Accuracy Score: %.2f" % scores, position=(0.0, 1.0))
plt.show()


joblib.dump(model, model_filename)

# SHAP 값 계산
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# SHAP Force Plot (개별 예측에 대한 설명)
shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])






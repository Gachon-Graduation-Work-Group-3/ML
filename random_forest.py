import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # 회귀 모델 사용
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import joblib

import warnings
warnings.filterwarnings('ignore')

#그래프 한글 폰트
matplotlib.rcParams['font.family'] = 'AppleGothic' 
matplotlib.rcParams['axes.unicode_minus'] = False 

model_filename = 'random_forest_model.pkl'

data = 'cars_processed.csv'
df = pd.read_csv(data, index_col=0)

corr = df.select_dtypes('number').corr()
sns.heatmap(corr)
plt.show()

print(corr)
price_corr = corr['가격']
#price_corr = price_corr[(price_corr > 0.1) | (price_corr < -0.1)]
price_corr = price_corr.sort_values()
print(price_corr)

X_columns = price_corr.index.to_numpy()
X = df[X_columns].drop(labels=['가격'], axis=1)
y = df['가격']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rfc = RandomForestRegressor(n_estimators=600, 
                            random_state=0, 
                            max_depth=10, 
                            min_samples_split=3,
                            min_samples_leaf=2,
                            max_features= None)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

joblib.dump(rfc, model_filename)

scores = rfc.score(X_test, y_test)
index = range(0, y_pred.size)
plt.plot(index, y_test, label='y_test', color='lightblue')
plt.plot(index, y_pred, label='y_pred', color='orange')
plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.1),
               ncol=2, fancybox=True, shadow=False)
plt.xlabel('index')
plt.ylabel('price (백만)')
plt.title("Accuracy Score: %.2f" % scores, position=(0.0, 1.0))
plt.show()

# 피처 중요도 가져오기
feature_importances = rfc.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 출력
print(importance_df)


mse = mean_squared_error(y_test, y_pred)

print(f"\n모델 성능 평가:\nMSE: {mse:.2f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # 회귀 모델 사용
import category_encoders as ce
import joblib

import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'AppleGothic'  # Mac에서 제공되는 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 처리

model_filename = 'random_forest_model.pkl'

data = 'cars_processed.csv'
df = pd.read_csv(data)

#print(df.shape)
#print(df.head())

x = df.drop(['가격'], axis=1)
y = df['가격']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

encoder = ce.OrdinalEncoder(cols=x.columns.to_list())

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

rfc = RandomForestRegressor(n_estimators=325, 
                            random_state=0, 
                            max_depth=10, 
                            min_samples_split=2,
                            min_samples_leaf=2,
                            max_features= None)
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

feature_scores = pd.Series(rfc.feature_importances_, index=x_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

joblib.dump(rfc, model_filename)




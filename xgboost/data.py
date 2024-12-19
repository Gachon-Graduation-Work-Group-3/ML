import pandas as pd
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'AppleGothic' 
plt.rcParams['axes.unicode_minus'] = False 

data = 'xgboost/data/cars_processed.csv'
df = pd.read_csv(data, index_col=0,)
df = df.select_dtypes('number')

correlation_with_price = df.corr()['신차대비가격'].sort_values(ascending=False)

# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,  annot_kws={"size": 5} )
plt.title("Correlation Heatmap with All Variables")
plt.xticks(fontsize=5)  # x축 글씨 크기
plt.yticks(fontsize=5)  # y축 글씨 크기
plt.show()


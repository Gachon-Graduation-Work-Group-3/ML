import pandas as pd
import numpy as np
from utils import changemodel, changemodelname, changeyear, engine, guar, clean_distance,parse_date
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sns

scaler = MinMaxScaler()

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

data = 'cars.csv'
df = pd.read_csv(data)

matchfuel={'디젤':0, 'LPG':1, '가솔린':2, '가솔린 하이브리드':3,'가솔린/LPG겸용':4 ,'전기':5, 'LPG 하이브리드':6, '수소':7, '가솔린/CNG겸용':8}
matchop={'무':0,'유':1}
insurinfo={'미등록':0,'등록':1}

#df = df.drop_duplicates(subset=['차량번호'], keep='first')

df = df[~df['신차대비가격'].isin(['소유', '반납', '운용', '렌터카'])]
df['신차대비가격'] = df['신차대비가격'].str.replace("%", "").astype(float) / 100
df['신차대비가격']=df['신차대비가격'].apply(lambda x: np.nan if x=='준비중' else x)
df = df.dropna(subset=['신차대비가격'])
df['최초등록일'] = df['최초등록일'].apply(parse_date)

#df['이름'] = df['이름'].apply(changemodel).apply(changemodelname)
df['연식'] = df['연식'].apply(changeyear).astype(float)
df = df.dropna(subset=['연식'])

df['주행거리'] = df['주행거리'].apply(clean_distance).astype(float)
df = df.dropna(subset=['주행거리'])
df = df[df['주행거리'] >= 10000]

df['배기량']=df['배기량'].apply(engine).astype(float)
df = df.dropna(subset=['배기량'])

df['연비'] = df['연비'].astype(str).replace(r'km/ℓ', '', regex=True).replace(r'km/kg', '', regex=True).str.replace(' ', '').replace({'': np.nan, '-': np.nan}).astype(float)
df = df.dropna(subset=['연비'])

df['최고출력'] = df['최고출력'].astype(str).replace(r'마력', '', regex=True).str.replace(' ', '').replace({'': np.nan, '-': np.nan}).astype(float)
df = df.dropna(subset=['최고출력'])

df['보증정보'] = df['보증정보'].apply(guar)
df['연료'] = df['연료'].map(matchfuel)


df=df.replace(matchop)
df=df.replace(insurinfo)


df['가격']=df['가격'].apply(lambda x:x.replace('만','').replace(',','')[:-1])
df = df[~df['가격'].isin(['[가격상담', '[계약', '[보류', '상담0000', '렌터카0000', '운용리스0000', '상담', '렌터카', '운용리스'])]
df = df[df['가격'].astype(int) <= 7000]


print(df['가격'].describe())
#df['가격'] = scaler.fit_transform(np.log10(df['가격'].astype(int)).values.reshape(-1, 1))



#필요없는 정보 삭제
df = df.drop(labels=['연료','보증정보','소유자변경','설명글', '링크', '색상','후측방경보','네비게이션(비순정)','자동주차','선루프','파노라마선루프','열선시트(앞좌석)','열선시트(뒷좌석)','동승석에어백','후방센서','전방센서','후방카메라','열선핸들','오토라이트','크루즈컨트롤','전손','침수전손','침수분손','도난','판금','부식', '내차피해_횟수', '내차피해_금액', '타차가해_횟수', '타차가해_금액', '최대토크', '교환', '보험처리수' ,'전방카메라', '어라운드뷰','네비게이션(순정)','불법구조변경', '사고침수유무'], axis=1)
df = df.fillna(0)
df.to_csv('cars_processed.csv')

na_check = df.isna().sum()

# inf 값이 있는지 확인 (양의 무한대, 음의 무한대)
inf_check = ((df == float('inf')) | (df == float('-inf'))).sum()

# 결과 출력
print("NaN 값의 개수:")
print(na_check)

print("\nInf 값의 개수:")
print(inf_check)



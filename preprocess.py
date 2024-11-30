import pandas as pd
import numpy as np
from utils import changemodel, changemodelname, changeyear, engine, guar, clean_distance,parse_date

data = 'cars.csv'
df = pd.read_csv(data)

matchfuel={'디젤':0, 'LPG':1, '가솔린':2, '가솔린 하이브리드':3,'가솔린/LPG겸용':4 ,'전기':5, 'LPG 하이브리드':6, '수소':7, '가솔린/CNG겸용':8}
matchop={'무':0,'유':1}
insurinfo={'미등록':0,'등록':1}

#df = df[~df['신차대비가격'].isin(['소유', '반납', '운용', '렌터카'])]
#df['신차대비가격'] = df['신차대비가격'].str.replace("%", "").astype(float) / 100
#df['신차대비가격']=df['신차대비가격'].apply(lambda x: np.nan if x=='준비중' else x)
df['최초등록일'] = df['최초등록일'].apply(parse_date)

#df['이름'] = df['이름'].apply(changemodel).apply(changemodelname)
df['연식'] = df['연식'].apply(changeyear).astype(float)
df['주행거리'] = df['주행거리'].apply(clean_distance).astype(float)
df['배기량']=df['배기량'].apply(engine).astype(float)
df['연비'] = df['연비'].astype(str).replace(r'km/ℓ', '', regex=True).replace(r'km/kg', '', regex=True).str.replace(' ', '').replace({'': np.nan, '-': np.nan}).astype(float)
df['최고출력'] = df['최고출력'].astype(str).replace(r'마력', '', regex=True).str.replace(' ', '').replace({'': np.nan, '-': np.nan}).astype(float)
df['최대토크'] = df['최대토크'].astype(str).replace(r'kg.m', '', regex=True).str.replace(' ', '').replace({'': np.nan, '-': np.nan}).astype(float)


df['소유자변경'] = df['소유자변경'].fillna('0').astype(str).replace(r'회', '', regex=True).astype(float)


#guartable=df['보증정보'].apply(guar)
#guartable.columns=['보증여부','보증기간','보증거리']
#df=pd.concat([df,guartable],axis=1)
#df=df.drop(['보증정보'],axis=1)

df=df.replace(matchop)
df=df.replace(insurinfo)

df['가격']=df['가격'].apply(lambda x:x.replace('만','0000').replace(',','')[:-1])
df = df[~df['가격'].isin(['[가격상담', '[계약', '[보류', '상담0000', '렌터카0000', '운용리스0000'])]
df = df[df['가격'].astype(int) <= 70000000]

#df['보험_내차피해(가격)'] = df['보험_내차피해(가격)'].replace(',', '', regex=True).fillna(0).astype(int)
#df['보험_타차피해(가격)'] = df['보험_타차피해(가격)'].replace(',', '', regex=True).fillna(0).astype(int)
#df['사고상세_타차가해(횟수)'] = df['사고상세_타차가해(횟수)'].fillna(0).astype(int)

df['내차피해_횟수'] = df['내차피해_횟수'].fillna('0').astype(str).replace(r'회', '', regex=True).astype(float)
df['타차가해_횟수'] = df['타차가해_횟수'].fillna('0').astype(str).replace(r'회', '', regex=True).astype(float)

df['내차피해_금액'] = (
    df['내차피해_금액']
    .fillna('0')  # NaN 값을 '0'으로 대체
    .astype(str)  # 문자열로 변환
    .str.replace('[^\d]', '', regex=True)  # 숫자가 아닌 모든 문자 제거
    .replace('', '0')  # 빈 문자열은 '0'으로 대체
    .astype(float)  # 정수형 변환
)

df['타차가해_금액'] = (
    df['타차가해_금액']
    .fillna('0')  # NaN 값을 '0'으로 대체
    .astype(str)  # 문자열로 변환
    .str.replace('[^\d]', '', regex=True)  # 숫자가 아닌 모든 문자 제거
    .replace('', '0')  # 빈 문자열은 '0'으로 대체
    .astype(float)  # 정수형 변환
)

#필요없는 정보 삭제
df = df.drop(labels=['설명글', '링크', '색상', '네비게이션(순정)','네비게이션(비순정)','자동주차','차량중량','선루프','파노라마선루프','열선시트(앞좌석)','열선시트(뒷좌석)','동승석에어백','후측방경보','후방센서','전방센서','후방카메라','전방카메라','어라운드뷰','열선핸들','오토라이트','크루즈컨트롤','전손','침수전손','침수분손','도난', '신차대비가격','판금','교환','부식','사고침수유무','불법구조변경'], axis=1)

df.to_csv('cars_processed.csv')


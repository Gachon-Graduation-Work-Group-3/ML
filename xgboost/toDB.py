import pandas as pd
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from utils import parse_age

data = 'xgboost/data/on_sale_cars_prep.csv'
df = pd.read_csv(data)

matchop={'무':0,'유':1}
df=df.replace(matchop)
df=df.fillna(0)

df['연식'] = df['연식'].apply(parse_age)
df['최초등록일'] = df['최초등록일'].apply(parse_age)

df=df.dropna()

new_df = pd.DataFrame({
    'name': df['이름'],  # 기존 name 컬럼 복사
    'price': df['가격'],  # 기존 price 컬럼 복사
    'cc': df['배기량'],
    'max_out': df['최고출력'],
    'mileage': df['주행거리'],
    'torque': df['최대토크'],
    'weight': df['차량중량'],
    'number': df['차량번호'],
    'engine': df['엔진형식'],
    'first_reg': df['최초등록일'],
    'fuel_eff': df['연비'],
    'fuel': df['연료'],
    'link': df['링크'],
    'image': df['이미지'].apply(
    lambda x: 'https:' + ast.literal_eval(x)[0] if isinstance(x, str) and x != "nan" else np.nan),
    'description': df['설명글'],
    'age': df['연식'],
    'insure': df['보험처리수'],
    'brand': df['브랜드'],
    'new_price': df['신차가격'],
    'sunroof': df['선루프'],
    'pano_sunroof': df['파노라마선루프'],
    'heat_front': df['열선시트(앞좌석)'],
    'heat_back': df['열선시트(뒷좌석)'],
    'pass_air': df['동승석에어백'],
    'rear_warn': df['후측방경보'],
    'rear_sensor': df['후방센서'],
    'front_sensor': df['전방센서'],
    'rear_camera': df['후방카메라'],
    'front_camera': df['전방카메라'],
    'around_view': df['어라운드뷰'],
    'auto_light': df['오토라이트'],
    'cruise_cont': df['크루즈컨트롤'],
    'auto_park': df['자동주차'],
    'navi_gen': df['네비게이션(순정)'],
    'navi_non': df['네비게이션(비순정)'],
    'view': df['조회수'],
    'color': df['색상'],
})



new_df.to_csv('xgboost/data/cars_db.csv')


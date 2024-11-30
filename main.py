from joblib import load
import pandas as pd

data = {
        "이름": "2019 기아 스포티지 더 볼드 1.6 디젤 2WD 럭셔리",
        "가격": 18990000,
        "차량번호": "34구0554",
        "최초등록일": None,
        "연식": 7.0,  
        "주행거리": 76000.0,
        "연료": "디젤",
        "배기량": 1598.0,
        "보증정보": "불가",
        "엔진형식": None,
        "연비": 15.8,
        "최고출력": 136.0,
        "최대토크": 35,
        "보험처리수": 0.0,
        "소유자변경": 0.0,
        "내차피해_횟수": 0.0,
        "내차피해_금액": 0.0,
        "타차가해_횟수": 0.0,
        "타차가해_금액": 0.0
    }
car_data_df = pd.DataFrame([data])

#data = 'cars_processed.csv'
#df = pd.read_csv(data, index_col=0)
#car_data_df = df.iloc[[10]]
# DataFr
car_data_df = car_data_df.fillna(0)
car_data_df = car_data_df.select_dtypes('number')

model = load('XGBoost_model.pkl')
feature_names = model.get_booster().feature_names
car_data_df = car_data_df[feature_names]
predictions = model.predict(car_data_df)    

print(predictions)

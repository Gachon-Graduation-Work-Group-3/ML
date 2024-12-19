from fastapi import FastAPI
import sys
sys.path.append('./xgboost')

from utils import changeyear, parse_date
from joblib import load
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:3000",  # React 개발 서버
]

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 허용할 Origin 목록
    allow_credentials=True,       # 인증 정보 (쿠키) 허용 여부
    allow_methods=["*"],          # 모든 HTTP 메서드 허용
    allow_headers=["*"],          # 모든 HTTP 헤더 허용
)

class Car(BaseModel):
    age : str
    km : float
    cc : float
    fuel_eff : float
    high_out : float
    date : str
    view : int
    new_price : int
    brand : int

model = load('model/XGBoost_model.pkl')
feature_names = model.get_booster().feature_names

@app.post("/price/prediction")
async def price_prediction(
    car : Car
):
    data = {
        "연식": changeyear(car.age), 
        "주행거리": car.km,
        "배기량": car.cc,
        "연비": car.fuel_eff,
        "최고출력": car.high_out,
        "최초등록일" : parse_date(car.date),
        "조회수" : car.view,
        "신차가격" : car.new_price,
        "브랜드": car.brand
    }
    car_data_df = pd.DataFrame([data])
    car_data_df = car_data_df[feature_names]

    predictions = model.predict(car_data_df)  
    predicted_price = float(predictions[0])

    return {"predicted_price": predicted_price*car.new_price}
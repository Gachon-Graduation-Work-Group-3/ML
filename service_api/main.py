from fastapi import FastAPI
import sys
sys.path.append('./xgboost')

from utils import changeyear, parse_date
from joblib import load
import pandas as pd
from pydantic import BaseModel

class Car(BaseModel):
    age : str
    km : float
    cc : float
    fuel_eff : float
    high_out : float
    date : str
    view : int
    new_price : int

app = FastAPI()
model = load('xgboost/model/XGBoost_model.pkl')
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
        "신차가격" : car.new_price
    }
    car_data_df = pd.DataFrame([data])
    car_data_df = car_data_df[feature_names]

    predictions = model.predict(car_data_df)  
    predicted_price = float(predictions[0])

    return {"predicted_price": predicted_price*car.new_price}
from fastapi import FastAPI
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
    comp : float

app = FastAPI()
model = load('XGBoost_model.pkl')
feature_names = model.get_booster().feature_names

@app.post("/price/prediction")
async def price_prediction(
    car : Car
):
    data = {
        "신차대비가격" :  car.comp,
        "연식": changeyear(car.age), 
        "주행거리": car.km,
        "배기량": car.cc,
        "연비": car.fuel_eff,
        "최고출력": car.high_out,
        "최초등록일" : parse_date(car.date)
    }
    car_data_df = pd.DataFrame([data])
    car_data_df = car_data_df[feature_names]

    predictions = model.predict(car_data_df)  
    predicted_price = float(predictions[0])

    return {"predicted_price": predicted_price}
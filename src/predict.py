import joblib
import pandas as pd

model = joblib.load("models/house_price_model.pkl")

house = pd.DataFrame([{
    "Area": 1800,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Age": 8
}])

print("Predicted price:", model.predict(house)[0])

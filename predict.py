import joblib
import pandas as pd

# Load the saved model and encoders
model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')

def preprocess(input_dict):
    df = pd.DataFrame([input_dict])
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    return df

def predict_score(input_dict):
    X = preprocess(input_dict)
    return float(model.predict(X)[0])
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('archive/student-mat.csv',sep=';')
encoders = {}

categorical_cols = [
    'school','sex','address','famsize','Pstatus','Mjob','Fjob',
    'reason','guardian','schoolsup','famsup','paid','activities',
    'nursery','higher','internet','romantic'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, "encoders.pkl")

X=df.drop('G3',axis=1)
y=df['G3']

model=LinearRegression()
model.fit(X,y)
joblib.dump(model,'model.pkl')
print("model saved")
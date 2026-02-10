# main.py
from http.client import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_score

app = FastAPI()

class StudentInput(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int
@app.post("/predict")
def predict(data: StudentInput):
    try:
        input_dict = data.dict()
        result = predict_score(input_dict)

        return {
            "predicted_G3": result
        }

    except ValueError as e:
        # Usually caused by unseen labels in LabelEncoder
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input value: {str(e)}"
        )

    except KeyError as e:
        # Missing column or encoder mismatch
        raise HTTPException(
            status_code=400,
            detail=f"Missing or incorrect field: {str(e)}"
        )

    except Exception as e:
        # Catch-all for unexpected issues
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please check input values."
        )

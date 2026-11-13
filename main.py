from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0.0, le=15.0)
    sepal_width: float = Field(gt=0.0, le=14.0)
    petal_length: float = Field(gt=0.0, le=14.0)
    petal_width: float = Field(gt=0.0, le=13.0)


class Prediction(BaseModel):
    model_version: str
    prediction: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures) -> Prediction:
    return Prediction(model_version="dummy", prediction="Iris pallida")

from fastapi import FastAPI, status, HTTPException
from schemas import Features, PredictionResponse
from predict import predict_data


app = FastAPI()


def build_features(model: Features):
    """Return ordered feature vector for the model.
    """
    received = model.dict()
    expected = list(Features.__fields__.keys())

    extra = set(received) - set(expected)
    if extra:
        raise HTTPException(status_code=400, detail={
            "error": "unexpected_fields",
            "unexpected": list(extra),
        })

    return [[received[f] for f in expected]]


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_wine(data: Features):
    """Receive named wine features and return the predicted class.
    """
    try:
        features = build_features(data)
        prediction = predict_data(features)
        return PredictionResponse(prediction=int(prediction[0]))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    

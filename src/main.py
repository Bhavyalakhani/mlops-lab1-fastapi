from fastapi import FastAPI, status, HTTPException
from schemas import WineFeatures, PredictionResponse
from predict import predict_data


app = FastAPI()


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_wine(data: WineFeatures):
    """Receive named wine features and return the predicted class.
    """
    try:
        feature_vector = [list(data.dict().values())]
        prediction = predict_data(feature_vector)
        return PredictionResponse(prediction=int(prediction[0]))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    

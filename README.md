# mlops-lab1-fastapi

A minimal FastAPI example that trains a scikit-learn model (Wine dataset)
and exposes a /predict endpoint.

Repository layout (important files)

- `src/` - application source. Key files:
  - `train.py` - trains a RandomForest model and saves `model/predict_wine_dataset_random_forest.pkl`.
  - `predict.py` - loads the saved model and exposes a prediction helper used by the API.
  - `main.py` - FastAPI app with `/predict` endpoint (expects named JSON features).
  - `schemas.py` - Pydantic models for request/response validation.
- `model/` - where the trained model file is written (created by `train.py`).
- `assets/sample_request.json` - example request body for the `/predict` endpoint.
- `fastapi_lab1_env/` - virtualenv included for convenience (optional to use).

Quick run (minimal steps)

1. Create and activate the virtual environment named `fastapi_lab1_mlops_env` (macOS / zsh). Use `python` or `python3` as per your command line. 

```bash
python3 -m venv fastapi_lab1_mlops_env

source fastapi_lab1_mlops_env/bin/activate

pip install -r requirements.txt
```

2. Change into `src/`, train the model (this creates `../model/predict_wine_dataset_random_forest.pkl`):

```bash
cd src
python train.py
```

You should see INFO logs showing where the model was saved and the test accuracy.

3. Start the API server from `src/`:

```bash
uvicorn main:app --reload
```

4. Test the endpoint using the included sample JSON.

Note: if you prefer a GUI, open the automatic docs at http://127.0.0.1:8000/docs and
copy-paste the contents of `assets/sample_request.json` into the request body to try the
`/predict` endpoint.

If you run the curl command from the `src/` directory (alternative):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d @../assets/sample_request.json
```


If you run the curl command from the repository root (recommended):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d @assets/sample_request.json
```

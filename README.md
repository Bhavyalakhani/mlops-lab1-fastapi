# mlops-lab1-fastapi

A minimal FastAPI example that trains a scikit-learn model (Wine dataset)
and exposes a /predict endpoint.

## Repository layout 

- `src/` - application source. Key files:
  - `train.py` - trains a RandomForest model and saves it to `model/predict_wine_dataset_random_forest.pkl`.
  - `predict.py` - loads the saved model and exposes a prediction helper used by the API.
  - `main.py` - FastAPI app with `/predict` endpoint (expects named JSON features).
  - `schemas.py` - Pydantic models for request/response validation.
- `model/` - where the trained model file is written (created by `train.py`).
- `assets/sample_request.json` - example request body for the `/predict` endpoint.

## Setup & Testing Endpoint

1. Create and activate the virtual environment named `fastapi_lab1_mlops_env` (macOS / zsh). Use `python` or `python3` as per your command line. 

```bash
python3 -m venv fastapi_lab1_mlops_env
```

```bash
source fastapi_lab1_mlops_env/bin/activate
```

```bash
pip install -r requirements.txt
```

2. Change into `src/`, train the model (this creates `../model/predict_wine_dataset_random_forest.pkl`):

```bash
cd src
```

```bash
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


If you run the curl command from the repository root:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d @assets/sample_request.json
```

5. Sample request & response

Request (same as `assets/sample_request.json`):

```json
{
	"alcohol": 13.24,
	"malic_acid": 2.59,
	"ash": 2.87,
	"alcalinity_of_ash": 21.0,
	"magnesium": 118.0,
	"total_phenols": 2.8,
	"flavanoids": 2.69,
	"nonflavanoid_phenols": 0.39,
	"proanthocyanins": 1.82,
	"color_intensity": 4.32,
	"hue": 1.04,
	"od280_od315_of_diluted_wines": 2.93,
	"proline": 735.0
}
```

Sample response (JSON):

```json
{
	"prediction": 0
}
```

6. Shutdown the server by entering (Control + C)
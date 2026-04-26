# Sign Language AI

Local sign-language hand landmark and letter prediction project.

## What This Project Does

- Collects hand landmark data from webcam images
- Trains a vowel classifier from landmark features
- Runs a local FastAPI backend for prediction
- Integrates with a C# WPF client through HTTP

## Project Structure

- `backend_api.py` - local FastAPI prediction backend
- `build.py` - build landmark CSV from labeled image folders
- `collect_vowel_data.py` - collect landmark samples from webcam
- `train.py` - train the classifier from the CSV dataset
- `validate.py` - validate dataset quality before training
- `live.py` - local webcam prediction demo in Python
- `setup.ps1` - install dependencies and create the virtual environment
- `requirements.txt` - Python dependencies
- `data/images/` - local labeled images for dataset building
- `data/raw/` - generated CSV datasets
- `models/` - trained model files

## Requirements

- Windows
- Python 3.13 or compatible
- PowerShell
- Webcam

## Quick Setup

1. Clone the repository.
2. Open PowerShell in the project folder.
3. Run:

```powershell
.\setup.ps1
```

This will:

- create `venv` if it does not exist
- activate the virtual environment
- upgrade `pip`
- install all dependencies from `requirements.txt`

## Manual Setup

If you want to do it step by step:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Building the Dataset From Images

Put your images into folders like this:

```text
data/images/A/
data/images/E/
data/images/I/
data/images/O/
data/images/U/
```

Then run:

```powershell
python build.py
```

That will generate or update:

- `data/raw/vowels_from_images_landmarks.csv`

If you add new images later, run `python build.py` again. It will only process new images.

If you need to rebuild everything from scratch:

```powershell
python build.py --rebuild
```

## Validate the Dataset

Before training, validate the CSV:

```powershell
python validate.py --csv data/raw/vowels_from_images_landmarks.csv
```

This checks:

- class balance
- invalid labels
- parse errors
- duplicate rows
- feature variation

## Train the Model

Train from the image-built CSV:

```powershell
python train.py
```

This reads:

- `data/raw/vowels_from_images_landmarks.csv`

And writes:

- `models/vowel_random_forest.joblib`

You can also train on a different CSV:

```powershell
python train.py --csv data/raw/vowels_from_images_landmarks.csv --model-out models/vowel_random_forest.joblib
```

## Run the Local Prediction Demo

To test the model in Python:

```powershell
python live.py
```

## Run the FastAPI Backend

Start the local backend:

```powershell
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

Open the API docs in your browser:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## FastAPI Prediction Endpoint

`POST /predict`

Form fields:

- `file` - image upload (`image/*`)
- `threshold` - confidence threshold, default `0.65`
- `smooth_window` - smoothing window, default `6`
- `include_landmarks` - `true` or `false`

Example response fields:

- `hand_detected`
- `label`
- `raw_label`
- `confidence`
- `probabilities`
- `handedness`
- `handedness_score`
- `landmarks`

## WPF Client Integration

Your C# WPF app can call the local backend at:

```text
http://127.0.0.1:8000
```

Use the reusable client class in your WPF project to:

- send a frame to `/predict`
- receive label and confidence
- optionally draw landmarks on a Canvas overlay

## GitHub / Version Control Notes

The repository is configured so that local-only files stay out of GitHub:

- `venv/` and `.venv/`
- `data/images/`
- `data/raw/*.csv`
- cache folders like `__pycache__/`

The trained model file can be committed if you want other users to run the backend immediately.

## Recommended Workflow

1. Add or collect images.
2. Run `python build.py`.
3. Run `python validate.py --csv data/raw/vowels_from_images_landmarks.csv`.
4. Run `python train.py`.
5. Start the backend with `python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000`.
6. Connect the WPF client.

## Notes

- Keep the backend local by binding to `127.0.0.1`.
- If you change the dataset, retrain the model.
- If you add new image folders, just rerun `python build.py`.

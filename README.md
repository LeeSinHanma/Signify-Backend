# Sign Language AI

Local sign-language hand landmark and letter prediction project.

## What This Project Does

- Collects hand landmark data from webcam images
- Trains a vowel classifier from landmark features
- Runs a local FastAPI backend for prediction
- Integrates with a C# WPF client through HTTP

## Requirements

- Windows
- Python 3.13 or compatible
- PowerShell
- Webcam

---

## Quick Start: Running the Backend with FastAPI

Follow these steps to get the backend running locally with Uvicorn.

### Step 1: Initial Setup (One-Time)

Open PowerShell in the project folder and run the automated setup:

```powershell
.\setup.ps1
```

This will:

- Create a virtual environment (`venv`)
- Activate the virtual environment
- Upgrade `pip`
- Install all dependencies from `requirements.txt`

### Step 2: Activate the Virtual Environment

Every time you start a new PowerShell session, activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` appear at the beginning of your PowerShell prompt, indicating the virtual environment is active.

### Step 3: Start the Backend Server

Run the FastAPI backend with Uvicorn:

```powershell
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Step 4: Verify the Backend is Running

Open your browser and visit:

- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health
- **Alternative Docs**: http://127.0.0.1:8000/redoc

### Step 5: Stop the Backend

To stop the server, press `Ctrl+C` in the PowerShell window.

---

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

---

## Manual Setup

If you prefer to set up the environment step by step instead of using `setup.ps1`:

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

To test the model in Python without the backend:

```powershell
python live.py
```

---

## FastAPI Backend API Reference

### Prediction Endpoint

**POST** `/predict`

Send an image to the backend for hand detection and vowel prediction.

**Form Fields:**

- `file` - image file upload (`image/*`)
- `threshold` - confidence threshold (default: `0.65`)
- `smooth_window` - smoothing window for predictions (default: `6`)
- `include_landmarks` - `true` to include hand landmarks in response, `false` to exclude

**Example Response:**

```json
{
  "hand_detected": true,
  "label": "A",
  "raw_label": "A",
  "confidence": 0.92,
  "probabilities": {
    "A": 0.92,
    "E": 0.05,
    "I": 0.02,
    "O": 0.01,
    "U": 0.00
  },
  "handedness": "Right",
  "handedness_score": 0.98,
  "landmarks": [
    {"x": 0.5, "y": 0.3},
    ...
  ]
}
```

**Response Fields:**

- `hand_detected` - boolean, whether a hand was detected
- `label` - predicted vowel letter
- `raw_label` - raw prediction label
- `confidence` - confidence score (0-1)
- `probabilities` - dict of all vowel probabilities
- `handedness` - "Right" or "Left"
- `handedness_score` - confidence in handedness detection
- `landmarks` - array of hand landmark coordinates (if requested)

---

## WPF Client Integration

Your C# WPF application can communicate with the local backend to get real-time hand gesture predictions.

### Backend URL

```
http://127.0.0.1:8000
```

### Basic Integration Example

Your WPF client can:

1. **Capture frames** from a camera or image source
2. **Send frames** to the `/predict` endpoint via HTTP POST
3. **Receive predictions** including label and confidence
4. **Display results** in your UI
5. **Optionally render landmarks** on a Canvas overlay using the returned coordinates

### Common Integration Pattern

```
1. User starts prediction in WPF app
2. App captures frame from webcam
3. App converts frame to image and POSTs to http://127.0.0.1:8000/predict
4. Backend processes frame and returns JSON response
5. App displays predicted vowel and confidence
6. Optionally draw landmarks on screen using returned coordinates
```

### Prerequisites for C# Integration

- Backend must be running (`python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000`)
- Your WPF application must have network access to `127.0.0.1:8000`
- Use a C# HTTP client (e.g., `HttpClient`) to send requests

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

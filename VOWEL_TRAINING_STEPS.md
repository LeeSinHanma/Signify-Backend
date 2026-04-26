Vowel Sign Training Workflow (A, E, I, O, U)

Data storage locations

- Raw training samples: data/raw/vowels_landmarks.csv
- Trained model: models/vowel_random_forest.joblib
- Data collection script: collect_vowel_data.py
- Image dataset builder: build_dataset_from_images.py
- Dataset validator: validate_dataset.py
- Training script: train_vowel_model.py
- Live inference script: live_vowel_predict.py

Image-based dataset storage (optional)

- Put images in label folders under data/images/
- Example: data/images/A, data/images/E, data/images/I, data/images/O, data/images/U
- Built landmark CSV from images: data/raw/vowels_from_images_landmarks.csv

1. Install required packages in your venv
   pip install mediapipe opencv-python scikit-learn joblib numpy

2. Collect data
   Run:
   python collect_vowel_data.py

While camera window is open:

- Show sign A and press A repeatedly
- Show sign E and press E repeatedly
- Show sign I and press I repeatedly
- Show sign O and press O repeatedly
- Show sign U and press U repeatedly
- Press Q to quit

Collection target

- Start with at least 200 samples per vowel
- Better: 500 to 1000 samples per vowel
- Keep classes balanced

Alternative: build dataset from labeled images

1. Place images in this structure:

- data/images/A/\*.jpg
- data/images/E/\*.jpg
- data/images/I/\*.jpg
- data/images/O/\*.jpg
- data/images/U/\*.jpg

2. Convert images to landmark CSV:

python build_dataset_from_images.py

This writes data/raw/vowels_from_images_landmarks.csv.

3. Validate dataset (recommended before training)
   Run:
   python validate_dataset.py

If you built CSV from images, run:

python validate_dataset.py --csv data/raw/vowels_from_images_landmarks.csv

Check for:

- Class balance across A/E/I/O/U
- Invalid labels or parse errors
- Too many duplicate rows
- Very low feature variation

4. Train model
   Run:
   python train_vowel_model.py

If you built CSV from images, run:

python train_vowel_model.py --csv data/raw/vowels_from_images_landmarks.csv --model-out models/vowel_from_images_random_forest.joblib

This will:

- Read data/raw/vowels_landmarks.csv
- Train a Random Forest classifier
- Print classification report and confusion matrix
- Save model to models/vowel_random_forest.joblib

5. Run live prediction
   Run:
   python live_vowel_predict.py

You should see:

- Predicted vowel label
- Confidence score

Quality tips

- Capture data in different lighting and backgrounds
- Capture slight rotation and distance changes
- Keep hand centered but varied
- If one vowel is weak in confusion matrix, collect more of that vowel

Next step after vowels

- Add consonants that are static first
- Save all labels in one larger CSV
- Retrain and evaluate again

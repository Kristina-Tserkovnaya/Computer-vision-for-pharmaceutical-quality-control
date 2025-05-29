# Computer-vision-for-pharmaceutical-quality-control

This project is part of a Master's thesis and shows how deep learning can be used to detect defects in pharmaceutical tablets and capsules.

The repository includes:
- Feature extraction using pretrained CNN models
- Fine-tuning of selected models
- A Streamlit web app to test model predictions on new images

## Files

app.py #streamlit app for live testing

requirements_app.txt # python libraries needed to run the app

models/convnext_base_all_data_best.pt #example pretrained model

features_extraction.ipynb #notebook for feature extraction

fine_tuning.ipynb #notebook for model training

## How to run the app

1. Install the required libraries:

```bash
pip install -r requirements_app.txt
```

2. Make sure the model file is in the models/ folder.

3. Start the app:

```bash
streamlit run app.py
```

4. The app will open in your browser. Upload images of pills to test predictions.

## Notes
The app supports both GPU and CPU.

Image filenames should include the true label (e.g., Proper, Proper_01.jpg, Broken, Broken_02.jpg).

Metrics like accuracy, precision, recall, and F1-score are shown after predictions.

# CardioScan AI

ECG arrhythmia detection research prototype built with a 1D residual CNN, Flask REST API, and React frontend.

## Project Overview

CardioScan AI processes MIT-BIH/WFDB ECG recordings, extracts individual beats, classifies them into five AAMI-oriented classes, and presents the analysis through a web interface.

The five classes are:

| Class | Meaning |
|---|---|
| N | Normal |
| S | Supraventricular ectopic |
| V | Ventricular ectopic |
| F | Fusion |
| Q | Unknown / unclassifiable |

The current training pipeline retains up to 20,000 beats per class from the MIT-BIH Arrhythmia Database. In the supplied training run, 39,332 beats were retained.

## Results

The supplied evaluation artifact reports:

- Test accuracy: 98.95%
- Macro F1: 96.91%
- Weighted F1: 98.96%
- Ventricular (V) F1: 0.98
- Fusion (F) F1: 0.90

Important: these results come from a stratified beat-level split. The split is not patient-independent, so the same patient records can contribute beats to training and test sets. The reported accuracy must therefore not be interpreted as evidence of clinical or patient-independent generalisation.

## Architecture

```text
MIT-BIH / WFDB ECG
        |
        v
Bandpass filtering (0.5–45 Hz)
        |
        v
R-peak detection
        |
        v
Beat segmentation
(200 ms before + 380 ms after)
        |
        v
Per-beat Z-score normalisation
        |
        v
1D Residual CNN
        |
        v
5-class softmax prediction
        |
        v
Flask REST API
        |
        v
React ECG analysis interface
```

The CNN uses approximately 530,373 trainable parameters.

## Repository Structure

```text
CardioScan/
├── backend/
│   ├── app.py
│   └── models/
│       ├── evaluation_report.json
│       └── training_history.json
├── docs/
│   ├── CardioScan_Project_Report.docx
│   ├── CardioScan_Presentation.pptx
│   └── CardioScan_Pipeline.pdf
├── ml/
│   ├── data/
│   │   └── metadata.json
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── data_preprocessing.py
│   └── train_model.py
├── src/
│   ├── components/
│   │   ├── ResultsView.jsx
│   │   └── UploadView.jsx
│   ├── App.css
│   ├── App.jsx
│   └── main.jsx
├── index.html
├── package.json
├── package-lock.json
├── requirements.txt
├── setup.bat
├── setup.sh
├── vite.config.js
├── .gitignore
└── LICENSE
```

Generated datasets, downloaded MIT-BIH signal files, TensorBoard logs, and `node_modules` are intentionally excluded from the repository. They can be recreated locally.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- Internet access for downloading the MIT-BIH database during preprocessing
- TensorFlow-compatible CPU/GPU environment

## Installation

### Windows

```bat
setup.bat
```

### macOS / Linux

```bash
bash setup.sh
```

The setup scripts install Python and frontend dependencies, download and preprocess the MIT-BIH records, and train the CNN.

If you prefer to run the steps manually:

```bash
pip install -r requirements.txt
npm install

cd ml
python data_preprocessing.py
python train_model.py
cd ..
```

Training creates:

```text
backend/models/arrhythmia_model.keras
backend/models/arrhythmia_model.pkl
backend/models/best_weights.weights.h5
backend/models/training_history.json
backend/models/evaluation_report.json
```

The trained model files are generated artifacts and are not included in this repository snapshot. The supplied project files include the evaluation artifacts from the reported training run.

## Running the Application

Start the Flask backend:

```bash
cd backend
python app.py
```

Start the React frontend in a second terminal:

```bash
npm run dev
```

Open the Vite URL shown in the terminal, normally:

```text
http://localhost:3000
```

For real inference, upload a matching `.dat` signal file and `.hea` header file from a WFDB record.

## Demo Mode

For frontend-only testing without a trained model:

```bash
CARDIOSCAN_DEMO=1 python backend/app.py
```

Demo mode returns synthetic demonstration data. It is not a model prediction and must not be used as a medical result.

## Dataset

The project uses the MIT-BIH Arrhythmia Database from PhysioNet.

The database contains 48 half-hour two-channel ambulatory ECG recordings from 47 subjects, sampled at 360 samples/second. The project uses the first channel (MLII) for the main processing path.

The dataset is not redistributed in this repository. Run:

```bash
cd ml
python data_preprocessing.py
```

to download it through `wfdb`.

## Medical Disclaimer

CardioScan AI is an academic research prototype. It has not been clinically validated and is not a medical device or a substitute for professional medical diagnosis. Model outputs require appropriate clinical review and should not be used to make patient-care decisions.

## Documentation

The `docs/` directory contains:

- Project report
- Project presentation
- End-to-end pipeline flowchart

The report has been cleaned to remove placeholders and to align the documented dataset statistics, model parameter count, evaluation protocol, and clinical claims with the supplied implementation.

## License

The project source code is released under the MIT License. The MIT-BIH Arrhythmia Database is a separate dataset with its own access and licensing terms; consult PhysioNet before redistributing or using the dataset.

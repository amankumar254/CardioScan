@echo off
setlocal

python -m venv .venv
call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
npm install

echo.
echo CardioScan setup complete.
echo Run: cd ml ^&^& python data_preprocessing.py
echo Then: python train_model.py
echo Then start backend and frontend in separate terminals.

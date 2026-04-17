# Ingredient Scanner

Simple Flask MVP for looking up a single ingredient, analyzing full ingredient labels, and uploading label photos for OCR-assisted review.

## Run locally

```powershell
cd "C:\Users\joeym\Repos\ingredient-scanner"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## OCR setup

Photo scanning now uses free browser-side Tesseract.js in production and locally.

That means:
- no Google Vision setup
- no OCR API key
- no OCR billing just to scan labels

## Local secrets

The `secrets` folder is still ignored by git, but OCR scanning no longer requires a local secret file.

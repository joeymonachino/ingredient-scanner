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

## Production OCR on Vercel

For production, use Google Cloud Vision so OCR works on the web and on mobile in Vercel.

### Vercel environment variable

In your Vercel project settings, add this environment variable:

- `No OCR API key is required now; browser OCR runs with Tesseract.js`

You can also use `GOOGLE_VISION_API_KEY`, but `No OCR API key is required now; browser OCR runs with Tesseract.js` is the preferred name.

### Vercel steps

1. Open your Vercel project.
2. Go to `Settings` -> `Environment Variables`.
3. Add:
   - Name: `No OCR API key is required now; browser OCR runs with Tesseract.js`
   - Value: your Google Cloud Vision API key
4. Save.
5. Redeploy the app.

When that key is present, uploaded label photos use Google Vision OCR in production.

## Local secrets

The app now also supports a local secrets folder so you can run the same OCR flow on your machine without exporting environment variables.

Create this file:

- [No local OCR secret file is required for scanning anymore](C:/Users/joeym/Repos/ingredient-scanner/secrets/No local OCR secret file is required for scanning anymore)

Put only the raw API key inside the file. No quotes, no extra text.

Example file contents:

```text
AIza...
```

You can also use:

- [google_vision_api_key.txt](C:/Users/joeym/Repos/ingredient-scanner/secrets/google_vision_api_key.txt)

The `secrets` folder is ignored by git, so the key stays local.

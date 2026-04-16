# Ingredient Scanner

Simple Flask MVP for looking up a single ingredient and returning:

- likely chemistry family
- likely source profile
- processing level
- shopper signal
- quick blurb with live reference enrichment when available

## Run locally

```powershell
cd "C:\Users\joeym\Documents\New project\ingredient-scanner"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

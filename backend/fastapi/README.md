# verity — fake news detector

Minimalist FastAPI backend + plain HTML frontend.

## structure

```
fakenews/
  backend/
    main.py          ← FastAPI app
  frontend/
    index.html       ← single-file frontend
  requirements.txt
```

## setup

```bash
pip install -r requirements.txt
```

## run backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

## run frontend

Open `frontend/index.html` directly in your browser,
or serve it with any static server:

```bash
cd frontend
python -m http.server 3000
# then visit http://localhost:3000
```

## api

POST /analyze
```json
{
  "text": "news article text here",
  "model": "phase1"   // or "phase2"
}
```

Response:
```json
{
  "label": "REAL",
  "prob_real": 0.772,
  "prob_fake": 0.228,
  "tokens": [
    { "word": "US", "tier": 0, "leading_space": false },
    { "word": "Centcom", "tier": 3, "leading_space": true },
    ...
  ]
}
```

Tier 0 = no highlight, 1-4 = weak → strong attribution.

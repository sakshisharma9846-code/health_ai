# Health Guide – Flask + Machine Learning + AI

A website with login, additional info (age, weight, height, sleep), 50 everyday health problems,
age-based medicine guidance, an ML symptom checker and an AI assistant.

## Run in VS Code
1. Open this folder in VS Code (File > Open Folder).
2. Open a terminal (Ctrl + `) and run:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   python app.py
   ```
3. Open http://127.0.0.1:5000 in your browser.

## Enable the AI assistant (optional)
Get an API key from console.anthropic.com and set it before running:
```
set ANTHROPIC_API_KEY=your-key          # Windows CMD   (Mac/Linux: export ANTHROPIC_API_KEY=your-key)
```
Without a key the chat still works in offline mode using the ML model.

## Project structure
- `app.py` – Flask routes, validation, age-group logic, AI chat endpoint
- `ml_model.py` – TF-IDF + Logistic Regression symptom matcher
- `data/problems.json` – 50 problems (causes, avoid, eat, treatment, medicine, lifestyle, keywords)
- `templates/`, `static/` – HTML pages and CSS

## How the ML part works (for your presentation)
Training sentences are generated from each problem’s symptom keywords. TF-IDF turns text into numbers and
Logistic Regression predicts the most likely problems with probabilities. The accuracy printed at start-up is
measured on synthetic data, so it is optimistic. To improve it, train on a real symptom dataset (for example from Kaggle).

## Disclaimer
Information only, not medical advice. Always consult a doctor.

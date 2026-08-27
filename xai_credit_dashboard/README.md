# XAI Rural Credit Risk Assessment — Decision Support Prototype

Django interface for the explainable credit risk model developed in the
accompanying analysis notebook. Each assessment returns a default probability
together with a SHAP explanation of the factors driving that specific result.

## Requirements

**Python 3.10 or newer.** Check with `python3 --version`. On macOS the
system Python is often 3.9, which is too old — install a current version from
python.org or via `brew install python@3.12`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python manage.py runserver
```

Open http://127.0.0.1:8000

No database or migrations are required. The application is stateless: it loads
the trained pipeline at startup and scores submitted applications in memory.

## The model

`model_artifacts/credit_risk_model.joblib` ships with this project. It is the
fitted pipeline exported by Section 12 of the analysis notebook — preprocessing,
resampling and classifier in a single object, so inference applies exactly the
transformations fitted during training.

To replace it with a newly trained model, re-run the notebook and copy the
contents of `model_artifacts/` over this directory.

| | |
|---|---|
| Algorithm | XGBoost (`max_depth=6`, `learning_rate=0.3`) |
| Training records | 29,926 |
| Bad-loan base rate | 11.10% |
| Validation F1 | 0.318 |
| Validation ROC-AUC | 0.724 |

## What the model uses

**Inputs:** facility amount, outstanding balance, instalment amount, arrears,
client settlement type, repayment mode. Four derived features (log transforms
and three burden ratios) are computed in `assessment/ml.py`.

**Deliberately excluded:**

- Gender and marital status — protected attributes, removed on fair-lending
  grounds and retained in the notebook only for auditing.
- Whether penalty compensation was charged — levied as a consequence of missed
  repayment, so unavailable at assessment time. Including it raised measured
  F1 by 0.047 but would not be obtainable in use.

## Routes

| Route | Purpose |
|---|---|
| `/` | Assessment form and result |
| `/about/` | Model provenance and design decisions |
| `/api/predict/` | JSON endpoint, POST |

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"investment_total":200000,"current_balance":100000,"install_size":16000,
       "due_payment":4000,"client_type":"Rural","repay_mode":"I"}'
```

## Deployment

Set `DJANGO_DEBUG=False` and supply `DJANGO_SECRET_KEY` and
`DJANGO_ALLOWED_HOSTS` before exposing this on any network.

```bash
python manage.py collectstatic --noinput
gunicorn config.wsgi:application
```

**Vercel will not work.** Vercel's Python runtime is serverless with a 250 MB
unzipped bundle limit; scikit-learn, XGBoost and SHAP together exceed it before
the model file is counted, and cold starts would reload the explainer on every
request. Use a container or VM host instead — Render, Railway, Fly.io and
PythonAnywhere all run this as-is with the `gunicorn` command above.

## Scope

Prototype decision-support tool for academic demonstration. Risk classifications
are indicative and intended to support, not replace, a credit officer's
judgement. The model identifies roughly half of loans that subsequently go bad;
it is not a loan origination decision system, and it uses no weather, crop yield
or agronomic data.

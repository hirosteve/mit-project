"""
Inference engine for the XAI credit risk assessment platform.

Loads the pipeline trained in the analysis notebook and produces, for each
submitted application, a default probability together with a SHAP-based
explanation of the factors driving that specific prediction.

The feature engineering here mirrors Section 4 of the notebook exactly. Any
divergence between the two would silently corrupt predictions, so the
transformations are kept deliberately simple and explicit.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from django.conf import settings

_LOCK = threading.Lock()

# Human-readable labels for the engineered feature names produced by the
# ColumnTransformer. Used so that explanations are legible to a loan officer
# rather than exposing internal column names.
FEATURE_LABELS = {
    "num__LOG_INVESTMENT_TOTAL": "Facility size",
    "num__LOG_ACCCURRENTBALANCE": "Outstanding balance",
    "num__LOG_INSTALL_SIZE": "Instalment amount",
    "num__LOG_DUE_PAYMENT": "Amount in arrears",
    "num__DEBT_SERVICE_RATIO": "Debt service ratio",
    "num__ARREARS_INTENSITY": "Arrears intensity",
    "num__BALANCE_UTILISATION": "Balance utilisation",
    "cat__CLIENT_TYPE_Rural": "Client type: Rural",
    "cat__CLIENT_TYPE_Semi-urban": "Client type: Semi-urban",
    "cat__CLIENT_TYPE_Urban": "Client type: Urban",
    "cat__REPAY_MODE_N": "Repayment mode: N",
    "cat__REPAY_MODE_I": "Repayment mode: I",
}


def humanise(name: str) -> str:
    """Convert an internal feature name into something a user can read."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    cleaned = name.replace("num__", "").replace("cat__", "").replace("LOG_", "")
    return cleaned.replace("_", " ").title()


# Parent variable for each engineered column. One-hot expansions of the same
# categorical variable share a parent so their contributions can be summed.
GROUP_LABELS = {
    "num__LOG_INVESTMENT_TOTAL": "Facility size",
    "num__LOG_ACCCURRENTBALANCE": "Outstanding balance",
    "num__LOG_INSTALL_SIZE": "Instalment amount",
    "num__LOG_DUE_PAYMENT": "Amount in arrears",
    "num__DEBT_SERVICE_RATIO": "Debt service ratio",
    "num__ARREARS_INTENSITY": "Arrears intensity",
    "num__BALANCE_UTILISATION": "Balance utilisation",
}


def group_of(name: str) -> str:
    """Map an engineered column to the variable a user would recognise."""
    if name in GROUP_LABELS:
        return GROUP_LABELS[name]
    if name.startswith("cat__CLIENT_TYPE"):
        return "Client settlement type"
    if name.startswith("cat__REPAY_MODE"):
        return "Repayment mode"
    return humanise(name)


def engineer_features(raw: dict) -> pd.DataFrame:
    """Build the model feature frame from raw application inputs.

    Mirrors Section 4.2 of the analysis notebook. The transformations are
    row-wise and involve no fitted parameters, so they are safe to apply here.
    """
    investment = float(raw["investment_total"])
    balance = float(raw["current_balance"])
    instalment = float(raw["install_size"])
    arrears = float(raw["due_payment"])

    denom = investment if investment > 0 else np.nan

    row = {
        "LOG_INVESTMENT_TOTAL": np.log1p(max(investment, 0)),
        "LOG_ACCCURRENTBALANCE": np.log1p(max(balance, 0)),
        "LOG_INSTALL_SIZE": np.log1p(max(instalment, 0)),
        "LOG_DUE_PAYMENT": np.log1p(max(arrears, 0)),
        "DEBT_SERVICE_RATIO": float(np.clip(instalment / denom, 0, 10)) if denom else 0.0,
        "ARREARS_INTENSITY": float(np.clip(arrears / denom, 0, 10)) if denom else 0.0,
        "BALANCE_UTILISATION": float(np.clip(balance / denom, 0, 10)) if denom else 0.0,
        "CLIENT_TYPE": raw["client_type"],
        "REPAY_MODE": raw["repay_mode"],
    }
    return pd.DataFrame([row])


BAND_CSS = {"A": "low", "B": "moderate", "C": "elevated", "D": "high"}


def apply_policy(probability: float, requested: float, policy: dict) -> dict:
    """Convert a default probability into an allocation recommendation.

    Implements the decision layer derived in Section 11 of the analysis
    notebook. Thresholds are cost-derived rather than conventional, and the
    policy modulates exposure rather than issuing a binary approve/decline,
    so that a moderate-risk borrower receives reduced capital rather than
    outright refusal.
    """
    for band in policy["bands"]:
        if band["lo"] <= probability < band["hi"]:
            break
    else:
        band = policy["bands"][-1]

    letter = band["band"].strip()[0]
    factor = band["factor"]
    return {
        "band": band["band"],
        "band_class": BAND_CSS.get(letter, "high"),
        "allocation_factor": factor,
        "allocation_pct": round(factor * 100),
        "recommended_amount": requested * factor,
        "requested_amount": requested,
        "withheld_amount": requested * (1 - factor),
        "action": band["action"],
        "approved": factor > 0,
    }


class CreditRiskEngine:
    """Wraps the trained pipeline and its SHAP explainer."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)

        model_path = self.artifacts_dir / "credit_risk_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run Section 12 of the "
                "analysis notebook and copy model_artifacts/ into the project root."
            )

        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["prep"]
        self.classifier = self.pipeline.named_steps["clf"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.explainer = shap.TreeExplainer(self.classifier)

        policy_path = self.artifacts_dir / "decision_policy.json"
        if not policy_path.exists():
            raise FileNotFoundError(
                f"Decision policy not found at {policy_path}. Run Section 11 of the "
                "analysis notebook and copy model_artifacts/ into the project root."
            )
        self.policy = json.loads(policy_path.read_text())

        meta_path = self.artifacts_dir / "model_metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        global_path = self.artifacts_dir / "shap_global_importance.csv"
        if global_path.exists():
            g = pd.read_csv(global_path).head(8)
            self.global_importance = [
                {"feature": humanise(r.Feature), "share": round(float(r.Share_), 2)
                 if hasattr(r, "Share_") else round(float(r[2]), 2)}
                for r in g.itertuples()
            ]
        else:
            self.global_importance = []

    def assess(self, raw: dict) -> dict:
        """Score one application and explain the result."""
        X = engineer_features(raw)

        probability = float(self.pipeline.predict_proba(X)[0, 1])
        decision = apply_policy(probability, float(raw["investment_total"]), self.policy)

        transformed = self.preprocessor.transform(X)
        shap_raw = self.explainer.shap_values(transformed)

        if isinstance(shap_raw, list):
            values = np.asarray(shap_raw[1] if len(shap_raw) > 1 else shap_raw[0])[0]
        elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
            values = shap_raw[0, :, 1]
        else:
            values = np.asarray(shap_raw)[0]

        base = self.explainer.expected_value
        base = float(np.ravel(base)[-1] if np.ndim(base) else base)

        # One-hot columns are aggregated back to their parent variable. Without
        # this, a user who selected "Rural" would see a contribution line for
        # "Client type: Urban", which is correct Shapley arithmetic but reads as
        # nonsense to a loan officer.
        grouped: dict[str, float] = {}
        for name, value in zip(self.feature_names, values):
            parent = group_of(name)
            grouped[parent] = grouped.get(parent, 0.0) + float(value)

        contributions = []
        for parent, value in grouped.items():
            if abs(value) < 1e-6:
                continue
            contributions.append({
                "feature": parent,
                "value": round(value, 4),
                "direction": "increases" if value > 0 else "reduces",
                "magnitude": abs(value),
            })
        contributions.sort(key=lambda c: c["magnitude"], reverse=True)
        top = contributions[:8]

        largest = max((c["magnitude"] for c in top), default=1.0) or 1.0
        for c in top:
            c["width"] = round(c["magnitude"] / largest * 100, 1)

        return {
            "probability": probability,
            "percentage": round(probability * 100, 2),
            "base_value": round(base, 4),
            "threshold": self.policy["threshold"],
            "cost_assumptions": self.policy["cost_assumptions"],
            **decision,
            "contributions": top,
            "increasing": [c for c in top if c["value"] > 0][:4],
            "reducing": [c for c in top if c["value"] < 0][:4],
            "model_name": self.metadata.get("selected_model", "Unknown"),
        }


@lru_cache(maxsize=1)
def get_engine() -> CreditRiskEngine:
    """Return the singleton engine, loading it on first use."""
    with _LOCK:
        return CreditRiskEngine(settings.MODEL_ARTIFACTS_DIR)

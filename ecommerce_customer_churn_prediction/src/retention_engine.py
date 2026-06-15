from typing import Dict


def get_risk_level(churn_probability: float) -> str:
    if churn_probability >= 0.8:
        return "High Risk"
    if churn_probability >= 0.5:
        return "Medium Risk"
    return "Low Risk"


def get_retention_recommendations(churn_probability: float) -> Dict[str, str]:
    if churn_probability >= 0.8:
        return {
            "Tier": "High Priority",
            "Action 1": "Offer 15% discount",
            "Action 2": "Assign priority customer support",
            "Action 3": "Send cashback coupon",
        }
    if churn_probability >= 0.5:
        return {
            "Tier": "Moderate Priority",
            "Action 1": "Award loyalty points",
            "Action 2": "Send personalized offers",
            "Action 3": "Promote seasonal bundles",
        }
    return {
        "Tier": "Low Priority",
        "Action 1": "Continue regular engagement",
        "Action 2": "Share app highlights",
        "Action 3": "Encourage next purchase",
    }


def build_customer_summary(inputs: Dict[str, float]) -> str:
    summary = [f"{key}: {value}" for key, value in inputs.items()]
    return " | ".join(summary)

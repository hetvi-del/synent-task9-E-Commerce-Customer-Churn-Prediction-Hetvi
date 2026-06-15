import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.feature_engineering import engineer_features
from src.retention_engine import build_customer_summary, get_risk_level, get_retention_recommendations

MODEL_PATH = "models/churn_model.pkl"


def load_model(path: str):
    return joblib.load(path)


def create_input_dataframe(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])


def create_prediction_gauge(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 50], "color": "#2ecc71"},
                    {"range": [50, 80], "color": "#f1c40f"},
                    {"range": [80, 100], "color": "#e74c3c"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    st.set_page_config(
        page_title="E-Commerce Customer Churn Prediction System",
        page_icon="📦",
        layout="wide",
    )

    st.title("E-Commerce Customer Churn Prediction System")
    st.markdown(
        "Use this dashboard to estimate churn risk, understand key customer values, and receive retention recommendations."
    )

    with st.sidebar.form(key="customer_input"):
        st.header("Customer Profile")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        ordercount = st.number_input("Order Count", min_value=0, max_value=200, value=8)
        satisfactionscore = st.slider("Satisfaction Score", min_value=1, max_value=5, value=4)
        cashbackamount = st.number_input("Cashback Amount", min_value=0.0, max_value=1000.0, value=24.0)
        warehousetohome = st.number_input("Warehouse To Home Distance (km)", min_value=0, max_value=100, value=12)
        complaint = st.selectbox("Complaint", ["No", "Yes"])
        daysincelastorder = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=15)
        hourspendonapp = st.number_input("Hours Spend On App", min_value=0.0, max_value=100.0, value=6.5)
        preferredlogindevice = st.selectbox("Preferred Login Device", ["Desktop", "Mobile", "Tablet"])
        citytier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        preferredpaymentmode = st.selectbox("Preferred Payment Mode", ["Credit Card", "Debit Card", "E-wallet", "COD"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        preferedordercat = st.selectbox(
            "Preferred Order Category", ["Electronics", "Fashion", "Groceries", "Home", "Beauty"]
        )
        maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        submitted = st.form_submit_button("Predict Churn")

    inputs = {
        "tenure": tenure,
        "ordercount": ordercount,
        "satisfactionscore": satisfactionscore,
        "cashbackamount": cashbackamount,
        "warehousetohome": warehousetohome,
        "complaint": complaint,
        "daysincelastorder": daysincelastorder,
        "hourspendonapp": hourspendonapp,
        "preferredlogindevice": preferredlogindevice,
        "citytier": citytier,
        "preferredpaymentmode": preferredpaymentmode,
        "gender": gender,
        "preferedordercat": preferedordercat,
        "maritalstatus": maritalstatus,
        "numberofdeviceregistered": 1,
        "numberofaddress": 1,
        "orderamounthikefromlastyear": 5.0,
        "couponused": 0,
    }

    if submitted:
        try:
            bundle = load_model(MODEL_PATH)
            model = bundle
            raw_df = create_input_dataframe(inputs)
            enriched_df = engineer_features(raw_df)
            prediction = model.predict(enriched_df)[0]
            probability = model.predict_proba(enriched_df)[0][1]
            risk_level = get_risk_level(probability)
            recommendations = get_retention_recommendations(probability)
            summary_text = build_customer_summary(inputs)

            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Prediction", "Churn" if prediction == 1 else "Stay")
            col2.metric("Churn Probability", f"{probability:.2%}")
            col3.metric("Risk Level", risk_level)

            st.plotly_chart(create_prediction_gauge(probability), use_container_width=True)

            st.subheader("Customer Summary")
            st.write(summary_text)

            st.subheader("Retention Recommendations")
            for label, action in recommendations.items():
                st.write(f"**{label}:** {action}")

            st.markdown("---")
            st.subheader("Feature Highlights")
            st.write(
                "The dashboard uses tenure, order frequency, customer value, engagement score, and recent activity to identify churn risk and boost retention."
            )
        except FileNotFoundError:
            st.error("Model file not found. Run the training pipeline first to generate models/churn_model.pkl.")

    st.sidebar.markdown("---")
    st.sidebar.write("Built for internship submission, portfolio presentation, and business use cases.")


if __name__ == "__main__":
    main()

# E-Commerce Customer Churn Prediction and Retention Recommendation System

## Problem Statement

This project predicts whether an e-commerce customer is likely to churn (leave the platform) and recommends retention actions to reduce customer attrition.

## Business Impact

- Identify customers with high churn risk.
- Increase customer lifetime value with targeted retention offers.
- Maximize marketing ROI by focusing on high-priority interventions.
- Improve customer satisfaction through personalized incentives.

## Dataset Description

The dataset includes customer behavior, transaction history, engagement metrics, and service feedback. Key columns include:

- `CustomerID`
- `Tenure`
- `PreferredLoginDevice`
- `CityTier`
- `WarehouseToHome`
- `PreferredPaymentMode`
- `Gender`
- `HourSpendOnApp`
- `NumberOfDeviceRegistered`
- `PreferedOrderCat`
- `SatisfactionScore`
- `MaritalStatus`
- `NumberOfAddress`
- `Complain`
- `OrderAmountHikeFromLastYear`
- `CouponUsed`
- `OrderCount`
- `DaySinceLastOrder`
- `CashbackAmount`
- `Churn`

## Project Architecture

- `dataset/` - contains the base churn dataset.
- `notebooks/` - includes exploratory analysis and visualization work.
- `src/` - project modules for cleaning, feature engineering, training, evaluation, and retention logic.
- `models/` - stores the trained model artifact.
- `app.py` - Streamlit application for interactive churn prediction.
- `requirements.txt` - Python dependencies.

## Features

- Data loading and cleaning
- Missing value handling with median/mode imputation
- Advanced feature engineering
- Pipeline-based preprocessing with scaling and encoding
- Multiple model training and comparison
- Model selection using ROC AUC
- Performance reporting and visualization
- Retention recommendation engine
- Streamlit dashboard for business use

## Installation

1. Clone or download the repository.
2. Create a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Prepare the dataset in `dataset/customer_churn.csv`.
2. Run training from the `src` folder or import modules in a Python script.
3. Launch the Streamlit app:

```bash
streamlit run app.py
```

## Screenshots

Add dashboard and EDA visuals to `screenshots/` for portfolio presentation.

## Future Improvements

- Add automated hyperparameter tuning.
- Integrate real-time customer stream processing.
- Add customer segmentation and uplift modeling.
- Implement model monitoring and drift detection.

## Technologies Used

- Python
- Pandas / NumPy
- Matplotlib / Seaborn / Plotly
- Scikit-Learn
- Joblib
- Streamlit

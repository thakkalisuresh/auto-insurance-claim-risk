"""
Auto Insurance Claim Risk — review-prioritization dashboard.

A stakeholder tool that wraps the team's final logistic-regression model
(TBANLT 520 CRISP-DM project). Enter an applicant's details and the app returns
the predicted claim probability and whether the policy crosses the recommended
0.09 review threshold. The model, feature engineering, and threshold match the
report and the executed notebook exactly.

The model is trained once, at startup, from Car_Insurance_Claim.csv sitting in
this same repository, so there is no saved-model file to version-match and
nothing to download. Training a logistic regression on 8,000 rows takes a few
seconds.

Run locally:   streamlit run app.py
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Constants that mirror the notebook -------------------------------------
CSV_PATH = "Car_Insurance_Claim.csv"
# Public Drive copy of the dataset, used when the CSV is not already beside the
# app (e.g. on Streamlit Community Cloud). Downloading a 1.3 MB CSV is instant.
CSV_DRIVE_ID = "1LjTDCga4V1mLCcBgg7KwveTdLOJQtaT0"
BUSINESS_THRESHOLD = 0.09          # cost-benefit optimum (report Section 5)
RANDOM_STATE = 42

NUMERIC_WITH_MISSING = ["credit_score", "annual_mileage"]
NUMERIC_SKEWED = ["speeding_violations", "duis", "past_accidents"]
BINARY = ["vehicle_ownership", "married", "children"]
PROTECTED = ["age", "gender", "race"]           # audit-only, dropped from model
NOMINAL = ["vehicle_type", "postal_code"]
ORDINAL = ["driving_experience", "education", "income", "vehicle_year"]
REDUNDANT_ENGINEERED = ["new_driver", "high_mileage",
                        "total_incidents", "incident_type_count"]


# --- Feature engineering (identical to notebook Section 5) -------------------
def add_engineered_features(X, mileage_threshold):
    X = X.copy()
    X["total_incidents"] = (X["speeding_violations"] + X["duis"]
                            + X["past_accidents"])
    X["incident_type_count"] = ((X["speeding_violations"] > 0).astype(int)
                                + (X["duis"] > 0).astype(int)
                                + (X["past_accidents"] > 0).astype(int))
    X["new_driver"] = (X["driving_experience"] == "0-9y").astype(int)
    X["high_mileage"] = (X["annual_mileage"] > mileage_threshold).astype(int)
    X["high_exposure_new_driver"] = X["new_driver"] * X["high_mileage"]
    X["older_high_mileage"] = ((X["vehicle_year"] == "before 2015")
                               & (X["high_mileage"] == 1)).astype(int)
    X["new_driver_no_ownership"] = X["new_driver"] * (1 - X["vehicle_ownership"])
    return X


def build_predictive_preprocessor():
    """The Section 4.7 ColumnTransformer plus the ablation-driven drop of the
    four redundant engineered features (notebook Section 8)."""
    numeric_missing = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler())])
    skewed = Pipeline([("power", PowerTransformer(method="yeo-johnson")),
                       ("scaler", StandardScaler())])
    categorical = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore",
                                                     drop="first",
                                                     sparse_output=False))])
    return ColumnTransformer(
        transformers=[
            ("numeric_missing", numeric_missing, NUMERIC_WITH_MISSING),
            ("missing_flags", MissingIndicator(features="missing-only",
                                               error_on_new=False),
             NUMERIC_WITH_MISSING),
            ("numeric_skewed", skewed, NUMERIC_SKEWED),
            ("categorical", categorical, NOMINAL + ORDINAL),
            ("binary", "passthrough", BINARY),
            ("protected_attributes", "drop", PROTECTED),
            ("redundant_engineered_features", "drop", REDUNDANT_ENGINEERED),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


@st.cache_resource(show_spinner="Training the model from the dataset…")
def load_and_train():
    if not os.path.exists(CSV_PATH):
        urllib.request.urlretrieve(
            f"https://drive.google.com/uc?export=download&id={CSV_DRIVE_ID}",
            CSV_PATH,
        )
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower() for c in df.columns]
    y = df["outcome"].astype(int)
    X = df.drop(columns=["id", "outcome"])
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    mileage_threshold = X_train["annual_mileage"].median()
    X_train = add_engineered_features(X_train, mileage_threshold)
    pipe = Pipeline([("prep", build_predictive_preprocessor()),
                     ("model", LogisticRegression(max_iter=1000,
                                                  random_state=RANDOM_STATE))])
    pipe.fit(X_train, y_train)
    return pipe, float(mileage_threshold), list(X_train.columns)


def score(pipe, mileage_threshold, columns, inputs):
    row = {c: np.nan for c in columns}
    row.update(inputs)
    frame = pd.DataFrame([row])
    frame = add_engineered_features(frame, mileage_threshold)
    frame = frame[columns]
    return float(pipe.predict_proba(frame)[:, 1][0])


# --- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Auto Insurance Claim Risk",
                   page_icon="🚗", layout="wide")

pipe, mileage_threshold, columns = load_and_train()

st.title("Auto Insurance Claim Risk")
st.caption("Review-prioritization tool for the underwriting team. It flags "
           "higher-risk policies for human review; it does not price policies "
           "or approve or deny applicants. Trained on a synthetic dataset, so "
           "effect directions are trustworthy while exact magnitudes are "
           "illustrative.")

with st.sidebar:
    st.header("Applicant details")
    experience = st.selectbox("Driving experience",
                              ["0-9y", "10-19y", "20-29y", "30y+"])
    credit_score = st.slider("Credit score", 0.0, 1.0, 0.52, 0.01)
    annual_mileage = st.slider("Annual mileage", 2000, 22000, 12000, 500)
    vehicle_year = st.selectbox("Vehicle year", ["before 2015", "after 2015"])
    vehicle_type = st.selectbox("Vehicle type", ["sedan", "sports car"])
    owns = st.checkbox("Owns the vehicle outright", value=True)
    income = st.selectbox("Income",
                          ["poverty", "working class", "middle class",
                           "upper class"])
    education = st.selectbox("Education", ["none", "high school", "university"])
    married = st.checkbox("Married", value=False)
    children = st.checkbox("Has children", value=False)
    postal_code = st.selectbox("Postal code", [10238, 32765, 92101, 21217])
    speeding = st.number_input("Speeding violations", 0, 25, 0)
    duis = st.number_input("DUIs", 0, 10, 0)
    past_accidents = st.number_input("Past accidents", 0, 20, 0)
    st.divider()
    st.caption("Age, gender, and race are deliberately excluded from the "
               "model and are not collected here.")

inputs = {
    "driving_experience": experience,
    "credit_score": credit_score,
    "annual_mileage": float(annual_mileage),
    "vehicle_year": vehicle_year,
    "vehicle_type": vehicle_type,
    "vehicle_ownership": 1.0 if owns else 0.0,
    "income": income,
    "education": education,
    "married": 1.0 if married else 0.0,
    "children": 1.0 if children else 0.0,
    "postal_code": postal_code,
    "speeding_violations": int(speeding),
    "duis": int(duis),
    "past_accidents": int(past_accidents),
    # Present so the ColumnTransformer can drop them, never used by the model.
    "age": "26-39", "gender": "female", "race": "majority",
}

proba = score(pipe, mileage_threshold, columns, inputs)
flagged = proba >= BUSINESS_THRESHOLD

left, right = st.columns([1, 1])
with left:
    st.subheader("Prediction")
    st.metric("Claim probability", f"{proba:.1%}")
    st.progress(min(proba, 1.0))
    if flagged:
        st.error(f"FLAG for review — at or above the "
                 f"{BUSINESS_THRESHOLD:.0%} threshold. Route to an underwriter.")
    else:
        st.success(f"Standard handling — below the "
                   f"{BUSINESS_THRESHOLD:.0%} threshold.")

with right:
    st.subheader("How to read this")
    st.markdown(
        f"""
- The **0.09 threshold** comes from the cost-benefit analysis: a missed claim
  costs far more than an unnecessary review, so the model deliberately flags
  broadly and catches about **96%** of claimants.
- A flag means *review*, not *deny*. Precision at this threshold is low by
  design, so **every flag goes to a human**.
- The strongest risk signals are **new-driver status** and **financing rather
  than owning** the vehicle; both compound.
"""
    )

st.divider()
st.caption("Fairness note: because driving experience tracks age, flag rates "
           "differ by age group even though age is never a model input. Older "
           "groups fall below the four-fifths rule at this threshold, which is "
           "why young-driver flags in particular carry mandatory human review "
           "(see report Section 7).")

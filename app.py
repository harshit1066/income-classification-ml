import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Income Classification App", layout="wide")

st.title("Income Classification using Machine Learning")
st.write("This app predicts whether income is >50K or <=50K using multiple ML models.")

model_name = st.selectbox(
    "Select a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_paths = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = joblib.load(model_paths[model_name])

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload the test file csv")
    st.stop()

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race",
    "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

df = pd.read_csv(
    uploaded_file,
    header=None,
    names=COLUMN_NAMES,
    skipinitialspace=True,
    skiprows=1
)


st.subheader("Dataset Preview")
st.dataframe(df.head())

df = df.replace("?", np.nan)
df = df.dropna()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

X = df.drop("income", axis=1)
y = df["income"]

y_pred = model.predict(X)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
else:
    auc = "Not Available"

st.subheader("Evaluation Metrics")

st.write("Accuracy:", accuracy_score(y, y_pred))
st.write("Precision:", precision_score(y, y_pred))
st.write("Recall:", recall_score(y, y_pred))
st.write("F1 Score:", f1_score(y, y_pred))
st.write("AUC Score:", auc)
st.write("MCC Score:", matthews_corrcoef(y, y_pred))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y, y_pred))

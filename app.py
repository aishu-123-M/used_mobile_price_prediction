
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Used Mobile Price Predictor", page_icon="📱", layout="wide")

BASE = Path(__file__).parent
MODEL_FILE = BASE / "used_mobile_price_model.pkl"
DATA_FILE = BASE / "used_mobile_prices.csv"

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
    return pd.read_csv(DATA_FILE)

bundle = load_model()
df = load_data()

# ---------- Header ----------
st.title("📱 Used Mobile Price Prediction")
st.caption("Machine Learning + Streamlit + Chatbot")

tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "🤖 Chatbot", "ℹ️ About"])

# ---------- Prediction ----------
with tab1:
    st.subheader("Enter mobile details")

    if bundle is None:
        st.error("Model file not found. Run the training/export cell in the Colab first and place used_mobile_price_model.pkl in this folder.")
    else:
        model = bundle["model"]
        features = bundle["features"]
        model_name = bundle.get("model_name", "Best trained model")

        # Use the dataset to create sensible input controls.
        if df is not None:
            clean_df = df.copy()
            clean_df.columns = (
                clean_df.columns.str.strip().str.lower()
                .str.replace(" ", "_").str.replace("-", "_")
            )
        else:
            clean_df = None

        input_data = {}
        left, right = st.columns(2)

        for idx, col in enumerate(features):
            container = left if idx % 2 == 0 else right

            if clean_df is not None and col in clean_df.columns:
                series = clean_df[col]
                if pd.api.types.is_numeric_dtype(series):
                    med = float(pd.to_numeric(series, errors="coerce").median())
                    minv = float(pd.to_numeric(series, errors="coerce").min())
                    maxv = float(pd.to_numeric(series, errors="coerce").max())
                    if not np.isfinite(med):
                        med = 0.0
                    if not np.isfinite(minv):
                        minv = med
                    if not np.isfinite(maxv) or maxv < minv:
                        maxv = med + 1.0

                    # Integer-like columns get a number input.
                    if float(med).is_integer() and float(minv).is_integer() and float(maxv).is_integer():
                        input_data[col] = container.number_input(
                            col.replace("_", " ").title(),
                            min_value=int(minv), max_value=int(maxv),
                            value=int(med), step=1
                        )
                    else:
                        input_data[col] = container.number_input(
                            col.replace("_", " ").title(),
                            min_value=float(minv), max_value=float(maxv),
                            value=float(med)
                        )
                else:
                    values = series.dropna().astype(str).unique().tolist()
                    values = sorted(values)
                    if len(values) > 100:
                        input_data[col] = container.text_input(
                            col.replace("_", " ").title(), value=values[0] if values else ""
                        )
                    else:
                        input_data[col] = container.selectbox(
                            col.replace("_", " ").title(),
                            values if values else [""]
                        )
            else:
                input_data[col] = container.text_input(col.replace("_", " ").title())

        if st.button("💰 Predict Used Price", type="primary", use_container_width=True):
            try:
                row = pd.DataFrame([input_data])
                row = row[features]
                prediction = float(model.predict(row)[0])
                st.success(f"### Estimated Used Mobile Price: ₹{prediction:,.0f}")
                st.info(f"Model used: **{model_name}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------- Chatbot ----------
with tab2:
    st.subheader("🤖 Mobile Price Assistant")
    st.write("Ask questions about the project, model, metrics, or used-mobile pricing.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content":
             "Hi! 👋 I'm your Used Mobile Price Assistant. Ask me about the prediction model, features, R², MAE, RMSE, or how the app works."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        q = question.lower()
        if any(x in q for x in ["r2", "r²", "r2 score"]):
            answer = "R² (R-squared) measures how much of the variation in used-mobile prices is explained by the model. A value closer to 1 is generally better."
        elif "mae" in q:
            answer = "MAE (Mean Absolute Error) is the average absolute difference between the actual and predicted prices. Lower is better."
        elif "rmse" in q:
            answer = "RMSE (Root Mean Squared Error) gives more weight to large prediction errors. Lower is better."
        elif "algorithm" in q or "model" in q:
            answer = f"The Colab trains Linear Regression, Decision Tree, Random Forest, and XGBoost, then selects the model with the highest R² score. The deployed model is: {bundle.get('model_name', 'best model') if bundle else 'not loaded'}."
        elif "feature" in q or "input" in q:
            answer = "The prediction form uses the same feature columns that were used during training, so the deployed app stays consistent with the Colab pipeline."
        elif "price" in q or "predict" in q or "resale" in q:
            answer = "The app takes the mobile's training features and passes them through the saved preprocessing + regression pipeline to estimate its used/resale price."
        elif "how" in q and "work" in q:
            answer = "The workflow is: dataset → cleaning → preprocessing → train multiple regression models → compare MAE/RMSE/R² → select the best R² model → save it with joblib → Streamlit loads it for live prediction."
        else:
            answer = "I can explain the model, algorithms, R², MAE, RMSE, input features, preprocessing, prediction workflow, and Streamlit deployment. Try asking one of those!"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ---------- About ----------
with tab3:
    st.subheader("Project Overview")
    st.markdown("""
**Problem:** Estimate the resale price of a used mobile phone from its available attributes.

**Machine Learning workflow**
1. Clean the dataset
2. Separate features and target
3. Handle numeric and categorical features
4. Train multiple regression models
5. Compare MAE, RMSE and R²
6. Select the model with the highest R²
7. Save the complete preprocessing + model pipeline
8. Serve predictions through Streamlit

The chatbot is built into the app and does not require an API key.
""")

    if bundle:
        st.metric("Deployed Model", bundle.get("model_name", "Unknown"))
        st.write("Target:", bundle.get("target", "Unknown"))
        st.write("Features:", ", ".join(bundle.get("features", [])))

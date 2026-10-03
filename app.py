import streamlit as st
import joblib
import numpy as np

# ---------------------------------------------------------
# Load the trained Linear Regression model
# (this model was saved with joblib.dump, so it must be
# loaded with joblib.load, not pickle.load)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("sales_prediction_model.pkl")
    return model

model = load_model()

# ---------------------------------------------------------
# App UI
# ---------------------------------------------------------
st.set_page_config(page_title="Sales Prediction App", page_icon="📈")

st.title("📈 Sales Prediction App")
st.write(
    "Enter your **YouTube** and **Facebook** advertising spend to predict **Sales**."
)

# Input fields
youtube = st.number_input("YouTube Advertising Spend", min_value=0.0, value=0.0, step=1.0)
facebook = st.number_input("Facebook Advertising Spend", min_value=0.0, value=0.0, step=1.0)

# Predict button
if st.button("Predict Sales"):
    input_data = np.array([[youtube, facebook]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

# ---------------------------------------------------------
# Optional: show raw input for debugging
# ---------------------------------------------------------
with st.expander("Show input data"):
    st.write({"YouTube": youtube, "Facebook": facebook})

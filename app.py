
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Website heading
st.title("🏠 House Price Predictor")
st.write(
    "Estimate a house price using your trained "
    "Linear Regression machine learning model."
)

st.divider()

# Input form
st.subheader("Enter House Details")

with st.form("house_details"):
    square_footage = st.number_input(
        "Square Footage",
        min_value=1,
        max_value=20000,
        value=1500
    )

    num_bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        max_value=20,
        value=3
    )

    num_bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=0,
        max_value=20,
        value=2
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2010
    )

    lot_size = st.number_input(
        "Lot Size",
        min_value=0.0,
        max_value=1000.0,
        value=2.0,
        step=0.1
    )

    garage_size = st.number_input(
        "Garage Size",
        min_value=0,
        max_value=10,
        value=1
    )

    neighborhood_quality = st.slider(
        "Neighborhood Quality",
        min_value=1,
        max_value=10,
        value=7
    )

    submitted = st.form_submit_button(
        "Predict House Price 🏡",
        use_container_width=True
    )

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        "Square_Footage": square_footage,
        "Num_Bedrooms": num_bedrooms,
        "Num_Bathrooms": num_bathrooms,
        "Year_Built": year_built,
        "Lot_Size": lot_size,
        "Garage_Size": garage_size,
        "Neighborhood_Quality": neighborhood_quality
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric(
            "Estimated House Price",
            f"{prediction:,.2f}"
        )
        st.caption(
            "This is a model estimate, not a guaranteed market price."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

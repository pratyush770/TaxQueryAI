import streamlit as st
from functions.logic import property_efficiency, collection_gap
import pandas as pd
from functions.prediction import train_prediction_model

st.set_page_config(  # sets the page configuration
    page_title="Predict Values",
    page_icon=":speech_balloon:"
)
st.title("Predict Values")  # displays title
col1, col2 = st.columns(2)  # divides into 2 columns
col3, col4 = st.columns(2)

with col1:  # column 1
    prediction = st.selectbox("Prediction Type", ("Tax Collection", "Tax Demand", "Property Efficiency", "Collection Gap"))
    prediction = prediction.lower()
with col2:
    city = st.selectbox("City", ("Chennai", "Solapur", "Pune", "Erode", "Thanjavur", "Jabalpur", "Tiruchirappalli"))
with col3:  # column 2
    year = st.number_input("Year", value=2019, min_value=2019, max_value=2030, placeholder=2019)
with col4:
    ptype = st.selectbox("Property Type", ("Residential", "Commercial"))

df = None  # load the dataset
if city:
    df_path = f"https://raw.githubusercontent.com/pratyush770/TaxQueryAI/master/datasets/transformed_data/Property-Tax-{city}.csv"
    df = pd.read_csv(df_path)  # load CSV from GitHub


def get_prediction_response(value: str, city: str, property_type: str, year: int, df: pd.DataFrame):  # for getting prediction response
    predict_tax = train_prediction_model(df, property_type)  # train the model for the given property type
    prediction = predict_tax(year) if year is not None and year >= 2019 else None  # get the prediction for the year
    # if prediction is available, return only the relevant prediction based on user query
    if prediction:
        if value == "tax collection":  # for tax collection
            return f"The predicted tax collection for {city} {property_type} in {year} is {prediction['predicted_collection']} Cr"
        elif value == "tax demand":  # for tax demand
            return f"The predicted tax demand for {city} {property_type} in {year} is {prediction['predicted_demand']} Cr"
        return None


if prediction == "property efficiency":  # if user selects property efficiency
    pe = property_efficiency(city, year, ptype, df)
    st.write(f"The predicted property efficiency for {city} {ptype} in {year} is {pe} Cr.")
elif prediction == "collection gap":  # if user selects collection gap
    cgap = collection_gap(city, year, ptype, df)
    st.write(f"The predicted collection gap for {city} {ptype} in {year} is {cgap} %.")
else:  # is user selects tax collection or demand
    value = get_prediction_response(prediction, city, ptype, year, df)
    st.write(value)




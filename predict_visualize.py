import streamlit as st
from functions.logic import property_efficiency, collection_gap
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import altair as alt

st.set_page_config(  # set page configuration
    page_title="Predict Values",
    page_icon=":speech_balloon:"
)

st.title("Predict Values")  # title of the page
col1, col2 = st.columns(2)  # divide into 2 columns
col3, col4 = st.columns(2)

with col1:  # column1
    prediction = st.selectbox("Prediction Type", ("Tax Collection", "Tax Demand", "Property Efficiency", "Collection Gap"))
    prediction = prediction.lower()  # converts the selected value into lowercase
with col2:
    city = st.selectbox("City", ("Chennai", "Solapur", "Pune", "Erode", "Thanjavur", "Jabalpur", "Tiruchirappalli"))
with col3:  # column2
    year = st.number_input("Year", value=2018, min_value=2018, max_value=2030, placeholder=2018)
with col4:
    ptype = st.selectbox("Property Type", ("Residential", "Commercial"))
predict_button = st.button("Predict", type="primary")

df = None
if city:
    df_path = f"https://raw.githubusercontent.com/pratyush770/TaxQueryAI/master/datasets/transformed_data/Property-Tax-{city}.csv"
    df = pd.read_csv(df_path)  # load the csv from github


def predict_values(df, property_type):  # function to predict historical and predicted value
    years = ["2013_14", "2014_15", "2015_16", "2016_17", "2017_18"]  # existing years in the dataset
    year_numbers = np.array([2013, 2014, 2015, 2016, 2017]).reshape(-1, 1)

    y_collection = df[[f'Tax_Collection_Cr_{y}_{property_type}' for y in years]].sum(axis=0).values  # adds all rows
    y_demand = df[[f'Tax_Demand_Cr_{y}_{property_type}' for y in years]].sum(axis=0).values

    model_collection = LinearRegression().fit(year_numbers, y_collection)  # train models
    model_demand = LinearRegression().fit(year_numbers, y_demand)

    def predict_tax(year):  # function to predict value
        if 2013 <= year <= 2017:  # for historical data
            year_str = f"{year}_{str(year + 1)[-2:]}"
            return {
                "predicted_collection": df[f'Tax_Collection_Cr_{year_str}_{property_type}'].sum(),
                "predicted_demand": df[f'Tax_Demand_Cr_{year_str}_{property_type}'].sum()
            }
        else:  # for predictions
            return {
                "predicted_collection": round(model_collection.predict([[year]])[0], 2),
                "predicted_demand": round(model_demand.predict([[year]])[0], 2)
            }

    return predict_tax


def get_prediction_response(value: str, city: str, property_type: str, year: int, df: pd.DataFrame):  # function to get the prediction response
    predict_tax = predict_values(df, property_type)
    prediction = predict_tax(year)  # get the prediction

    if prediction:
        if value == "tax collection":  # for tax collection
            return prediction["predicted_collection"]
        elif value == "tax demand":  # for tax demand
            return prediction["predicted_demand"]
    return None


if predict_button:  # if clicked on button
    historical_years = list(range(2013, 2018))
    predictions = []  # empty list to store predictions

    if prediction in ["property efficiency", "collection gap"]:
        for y in historical_years:
            if prediction == "property efficiency":  # for property efficiency
                predictions.append(property_efficiency(city, y, ptype, df))  # append the values in the list
            else:  # for collection gap
                predictions.append(collection_gap(city, y, ptype, df))

        selected_year_value = property_efficiency(city, year, ptype, df) if prediction == "property efficiency" else collection_gap(city, year, ptype, df)
    else:  # for tax demand or collection
        for y in historical_years:
            predictions.append(get_prediction_response(prediction, city, ptype, y, df))

        selected_year_value = get_prediction_response(prediction, city, ptype, year, df)

    all_years = historical_years + [year]  # join all the years in a list
    all_values = predictions + [selected_year_value]  # get all the predictions
    all_types = ["Historical" for _ in historical_years] + ["Predicted"]

    chart_data = pd.DataFrame({"Year": all_years, "Value": all_values, "Type": all_types})  # convert to dataFrame for altair
    chart = (  # create altair chart with different colors
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x="Year:O",  # discrete year values
            y="Value:Q",
            color="Type:N",  # different color for historical & predicted
            tooltip=["Year", "Value", "Type"]
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart, use_container_width=True)  # display the chart

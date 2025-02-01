import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

tax_data_path = 'D:/TaxQueryAI/datasets/transformed_data/'  # path
cities = ['Pune', 'Solapur', 'Erode', 'Jabalpur', 'Thanjavur', 'Chennai', 'Tiruchirappalli']  # city list
models = {}

for city in cities:  # train models for each city and property type
    df = pd.read_csv(f'{tax_data_path}Property-Tax-{city}.csv')

    tax_sums = {  # sum tax collection & demand across years for both property types
        "collection": {
            "commercial": df[[
                'Tax_Collection_Cr_2013_14_Commercial',
                'Tax_Collection_Cr_2014_15_Commercial',
                'Tax_Collection_Cr_2015_16_Commercial',
                'Tax_Collection_Cr_2016_17_Commercial',
                'Tax_Collection_Cr_2017_18_Commercial'
            ]].sum(axis=0),

            "residential": df[[
                'Tax_Collection_Cr_2013_14_Residential',
                'Tax_Collection_Cr_2014_15_Residential',
                'Tax_Collection_Cr_2015_16_Residential',
                'Tax_Collection_Cr_2016_17_Residential',
                'Tax_Collection_Cr_2017_18_Residential'
            ]].sum(axis=0),
        },
        "demand": {
            "commercial": df[[
                'Tax_Demand_Cr_2013_14_Commercial',
                'Tax_Demand_Cr_2014_15_Commercial',
                'Tax_Demand_Cr_2015_16_Commercial',
                'Tax_Demand_Cr_2016_17_Commercial',
                'Tax_Demand_Cr_2017_18_Commercial'
            ]].sum(axis=0),

            "residential": df[[
                'Tax_Demand_Cr_2013_14_Residential',
                'Tax_Demand_Cr_2014_15_Residential',
                'Tax_Demand_Cr_2015_16_Residential',
                'Tax_Demand_Cr_2016_17_Residential',
                'Tax_Demand_Cr_2017_18_Residential'
            ]].sum(axis=0),
        }
    }

    X = np.array([2014, 2015, 2016, 2017, 2018]).reshape(-1, 1)
    city_models = {"collection": {}, "demand": {}}

    for tax_type in ["collection", "demand"]:
        for property_type in ["commercial", "residential"]:
            y = tax_sums[tax_type][property_type].values
            model = LinearRegression().fit(X, y)
            city_models[tax_type][property_type] = model

    models[city] = city_models


def predict_tax(city, year, tax_type, property_type):  # function to predict tax collection or demand
    city = city.title()
    property_type = property_type.lower()

    if city not in models:  # if city does not exist in the database
        return f"City '{city}' not found in the dataset."
    if property_type not in ["residential", "commercial"]:
        return f"Invalid property type '{property_type}'. Choose 'residential' or 'commercial'."
    if year < 2019:
        return None  # no prediction for past years

    predicted_tax = models[city][tax_type][property_type].predict([[year]])
    return round(predicted_tax[0], 2)


if __name__ == "__main__":  # example usage
    print(predict_tax("Pune", 2019, "demand", "residential"))
    print(predict_tax("Mumbai", 2023, "collection", "commercial"))

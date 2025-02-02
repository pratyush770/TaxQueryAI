import numpy as np
from sklearn.linear_model import LinearRegression


def train_prediction_model(df, property_type):  # train and predict
    tax_sums = {  # adds all the rows
        "collection": df[[f'Tax_Collection_Cr_2013_14_{property_type}',
                          f'Tax_Collection_Cr_2014_15_{property_type}',
                          f'Tax_Collection_Cr_2015_16_{property_type}',
                          f'Tax_Collection_Cr_2016_17_{property_type}',
                          f'Tax_Collection_Cr_2017_18_{property_type}']].sum(axis=0),

        "demand": df[[f'Tax_Demand_Cr_2013_14_{property_type}',
                      f'Tax_Demand_Cr_2014_15_{property_type}',
                      f'Tax_Demand_Cr_2015_16_{property_type}',
                      f'Tax_Demand_Cr_2016_17_{property_type}',
                      f'Tax_Demand_Cr_2017_18_{property_type}']].sum(axis=0)
    }

    X = np.array([2014, 2015, 2016, 2017, 2018]).reshape(-1, 1)
    models = {}

    for key in ["collection", "demand"]:
        y = tax_sums[key].values
        model = LinearRegression()  # LinearRegression() call
        model.fit(X, y)  # fits the model
        models[key] = model

    def predict_tax(year):  # function to predict tax collection and demand
        if year is not None and year >= 2019:  # since we already have data from 2013-18
            return {
                "predicted_collection": round(models["collection"].predict([[year]])[0], 2),  # for tax collection prediction
                "predicted_demand": round(models["demand"].predict([[year]])[0], 2)  # for tax demand prediction
            }
        return None

    return predict_tax

from secret_key import sec_key, mysql_uri_local
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser  # for parsing the output into a string
from langchain_core.runnables import RunnablePassthrough  # allows the function to pass as a runnable
from langchain_groq import ChatGroq
from functions.prediction import train_prediction_model
from langchain_core.prompts import ChatPromptTemplate
import re

model_name = "qwen-2.5-32b"  # name of model used
llm = ChatGroq(
    model_name=model_name,
    temperature=0.1,  # more accurate results
    groq_api_key=sec_key
)

SCHEMA_CACHE = None  # global schema cache


def get_schema(db: SQLDatabase):
    global SCHEMA_CACHE
    if SCHEMA_CACHE is None:
        SCHEMA_CACHE = db.get_table_info()
    return SCHEMA_CACHE


def get_sql_chain(db: SQLDatabase):  # function to get sql query
    template = """
        Based on the table schema below, write only a SQL query that would answer the user's question.
        Don't provide any extra information other than the sql query.
        {schema}
        Conversation History: {chat_history}

        For example: 
        Question: "How many rows are there in the pune table?"
        SQL Query: SELECT COUNT(Ward_Name) AS ward_count FROM pune;
        Question: What was the total property tax collection in 2013-14 residential for aundh in pune city?
        SQL Query: SELECT SUM(Tax_Collection_Cr_2013_14_Residential) AS total_tax_collected FROM pune WHERE Ward_Name = "Aundh";
        Question: What was the total tax demand in 2015-16 commercial for erode in zone 3?
        SQL Query: SELECT SUM(Tax_Demand_Cr_2015_16_Commercial) AS total_tax_demand FROM erode WHERE Zone_Name = "Zone-3";
        Question: What was the property efficiency for the year 2015-16 commercial for Chennai?
        SQL Query: SELECT ROUND((SUM(Tax_Collection_Cr_2015_16_Commercial) / SUM(Tax_Demand_Cr_2015_16_Commercial)) * 100, 2) AS property_efficiency_percent FROM chennai;
        Question: What was the property efficiency for pune from 2013-18 commercial?
        SQL Query: SELECT ROUND((SUM(Tax_Collection_Cr_2013_14_Commercial) + SUM(Tax_Collection_Cr_2014_15_Commercial) + SUM(Tax_Collection_Cr_2015_16_Commercial) + SUM(Tax_Collection_Cr_2016_17_Commercial) + SUM(Tax_Collection_Cr_2017_18_Commercial)) / (SUM(Tax_Demand_Cr_2013_14_Commercial) + SUM(Tax_Demand_Cr_2014_15_Commercial) + SUM(Tax_Demand_Cr_2015_16_Commercial) + SUM(Tax_Demand_Cr_2016_17_Commercial) + SUM(Tax_Demand_Cr_2017_18_Commercial)) * 100, 2) AS property_efficiency_percent FROM pune;
        Question: What was the collection gap for the year 2016-17 residential for Thanjavur?
        SQL Query: SELECT ROUND((SUM(Tax_Demand_Cr_2016_17_Residential) - SUM(Tax_Collection_Cr_2016_17_Residential)), 2) AS collection_gap FROM thanjvaur;
        Question: What was the collection gap for solapur from 2013-18 residential?
        SQL Query: SELECT ROUND((SUM(Tax_Demand_Cr_2013_14_Residential) + SUM(Tax_Demand_Cr_2014_15_Residential) + SUM(Tax_Demand_Cr_2015_16_Residential) + SUM(Tax_Demand_Cr_2016_17_Residential) + SUM(Tax_Demand_Cr_2017_18_Residential)) - (SUM(Tax_Collection_Cr_2013_14_Residential) + SUM(Tax_Collection_Cr_2014_15_Residential) + SUM(Tax_Collection_Cr_2015_16_Residential) + SUM(Tax_Collection_Cr_2016_17_Residential) + SUM(Tax_Collection_Cr_2017_18_Residential)), 2) AS collection_gap FROM solapur;

        Your turn:
        Question: {question}
        SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):  # function to get schema info
        return db.get_table_info()

    return (
            RunnablePassthrough.assign(schema=lambda _: get_schema(db))  # use cached schema
            | prompt
            | llm.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
    )


def get_prediction_response(user_query: str, city: str, property_type: str, year: int, df: pd.DataFrame):
    predict_tax = train_prediction_model(df, property_type)  # train the model for the given property type
    prediction = predict_tax(year) if year is not None and year >= 2019 else None  # get the prediction for the year
    # if prediction is available, return only the relevant prediction based on user query
    if prediction:
        if "efficiency" in user_query.lower():  # check for property efficiency query
            efficiency = round((prediction['predicted_collection'] / prediction['predicted_demand']) * 100, 2)
            return f"The predicted property efficiency for {city} {property_type} in {year} is {efficiency}%"
        elif "demand" in user_query.lower():
            return f"The predicted tax demand for {city} {property_type} in {year} is {prediction['predicted_demand']} Cr"
        elif "collection" in user_query.lower():
            return f"The predicted tax collection for {city} {property_type} in {year} is {prediction['predicted_collection']} Cr"
        else:
            return "Please specify whether you want the tax collection or demand prediction."
    return None


def get_sql_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)  # sql chain for other queries
    template = """
       Based on the table schema below, question, SQL query, and SQL response, write only a natural language response to the user's question.
       {schema}
       Conversation History: {chat_history}

       Question: {question}
       SQL Query: {query}
       SQL Response: {response}
       Natural Language Response (check if response is related to a property efficiency or counting entries, if neither, append " crore" at the end):
    """
    prompt = ChatPromptTemplate.from_template(template)  # another template to get the natural language response

    chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: get_schema(db),  # use cached schema
                response=lambda var: db.run(var["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


def collection_gap(city, year, property_type, df):  # for predicting collection gap
    predict_tax = train_prediction_model(df, property_type)  # train the prediction model for the given property type
    tax_demand = predict_tax(year)["predicted_demand"]  # get the prediction for tax demand
    tax_collection = predict_tax(year)["predicted_collection"]  # get the prediction for tax collection
    collection_gap = round((tax_demand - tax_collection), 2)  # calculate the collection gap
    return collection_gap


def property_efficiency(city, year, property_type, df):  # for predicting property efficiency
    predict_tax = train_prediction_model(df, property_type)  # train the prediction model for the given property type
    tax_demand = predict_tax(year)["predicted_demand"]  # get the prediction for tax demand
    tax_collection = predict_tax(year)["predicted_collection"]  # get the prediction for tax collection
    property_efficiency = round(((tax_collection / tax_demand) * 100), 2)  # calculate the property efficiency
    return property_efficiency


def extract_query_info(user_query):  # function to extract city, property type, and year
    cities = ['Pune', 'Solapur', 'Erode', 'Jabalpur', 'Thanjavur', 'Chennai', 'Tiruchirappalli']
    property_types = ["Residential", "Commercial"]
    city = next((c for c in cities if c.lower() in user_query.lower()), None)
    property_type = next((p for p in property_types if p.lower() in user_query.lower()), "Residential")
    # use regular expression to find a year between 2013 and 2050
    year_match = re.search(r'\b(201[3-9]|20[2-4][0-9]|2050)\b', user_query)
    year = int(year_match.group()) if year_match else None
    return city, property_type, year


def get_response(user_query: str, db: SQLDatabase, chat_history: list, city: str, property_type: str, year: int,
                 df: pd.DataFrame):
    # check for a prediction-based response
    prediction_response = get_prediction_response(user_query, city, property_type, year, df)
    if prediction_response:
        return prediction_response  # if we have a prediction, return it immediately
    # handle collection gap queries
    if "collection gap" in user_query.lower() and year > 2018:
        gap = collection_gap(city, year, property_type, df)
        return f"The predicted collection gap for {city} {property_type} in {year} is {gap} Cr"
    # handle property efficiency queries
    if "property efficiency" in user_query.lower() and year > 2018:
        efficiency = property_efficiency(city, year, property_type, df)
        return f"The predicted property efficiency for {city} {property_type} in {year} is {efficiency}%"
    # if no predictions applied, execute a normal SQL query
    return get_sql_response(user_query, db, chat_history)


if __name__ == "__main__":  # example usage
    mysql_uri = mysql_uri_local  # taking local db
    db = SQLDatabase.from_uri(mysql_uri)
    df = pd.read_csv("D:/TaxQueryAI/datasets/transformed_data/Property-Tax-Pune.csv")  # load the tax data as an example
    # For prediction
    user_query = "What will be the tax demand for the year 2019 in Pune for residential?"
    chat_history = []
    city = "Pune"
    property_type = "Residential"
    year = 2019
    response = get_response(user_query, db, chat_history, city, property_type, year, df)
    print(response)
    # For prediction (collection gap)
    user_query = "What will be the property efficiency (residential) for the year 2019 in Pune?"
    chat_history = []
    city = "Pune"
    property_type = "Residential"
    year = 2019
    response = get_response(user_query, db, chat_history, city, property_type, year, df)
    print(response)
    # For normal response
    user_query = "What was the total property tax collection in 2013-14 residential for aundh in pune city?"
    chat_history = []
    city = "Pune"
    property_type = "Residential"
    year = 2014
    response = get_response(user_query, db, chat_history, city, property_type, year, df)
    print(response)


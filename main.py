import os
from secret_key import sec_key
import streamlit as st
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from functions.logic import get_response, get_sql_chain
import re

os.environ['GROQ_API_KEY'] = sec_key  # set environment variable

if "db" not in st.session_state:  # initialize database connection
    mysql_uri = "mysql+mysqlconnector://root_readonly:matsumoto@localhost:3307/property_tax"
    st.session_state.db = SQLDatabase.from_uri(mysql_uri)

if "chat_history" not in st.session_state:  # initialize chat history
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm a SQL assistant. Ask me anything about your database.")
    ]

st.set_page_config(  # set page config
    page_title="TaxQueryAI",
    page_icon=":speech_balloon:"
)
st.title("MySQL Database Q&A Tool")

for message in st.session_state.chat_history:  # display chat history
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.markdown(message.content)

user_query = st.chat_input("Type a message...")  # asks for user input


def extract_query_info(user_query):  # function to extract city, property type, and year
    cities = ['Pune', 'Solapur', 'Erode', 'Jabalpur', 'Thanjavur', 'Chennai', 'Tiruchirappalli']
    property_types = ["Residential", "Commercial"]
    city = next((c for c in cities if c.lower() in user_query.lower()), None)
    property_type = next((p for p in property_types if p.lower() in user_query.lower()), "Residential")
    # use regular expression to find a year between 2013 and 2050
    year_match = re.search(r'\b(201[3-9]|20[2-4][0-9]|2050)\b', user_query)
    year = int(year_match.group()) if year_match else None
    return city, property_type, year


# edge case handling
def handle_edge_cases(user_query):
    polite_messages = ["thanks", "thank you", "thx", "appreciate it", "ty", "okay thanks", "thnx", "okay thank you"]
    welcome_messages = ["hi", "hello", "how are you?", "hey", "hey there"]
    if user_query.lower().strip() in welcome_messages:  # for welcome messages
        return "Hey! How's it going?"

    elif user_query.lower().strip() in polite_messages:  # for polite messages
        return "You're welcome! Let me know if you have any more questions."

    elif any(phrase in user_query.lower() for phrase in ["give me the sql query", "give me the query"]):  # for getting sql query if asked by user
        last_query = next(
            (msg.content for msg in reversed(st.session_state.chat_history) if isinstance(msg, HumanMessage)), None)
        if last_query:
            sql_chain = get_sql_chain(st.session_state.db)
            return sql_chain.invoke({"question": last_query, "chat_history": st.session_state.chat_history})
        else:
            return "I couldn't find a previous query to generate SQL for."

    city_keywords = {"cities", "tables", "database", "available", "names"}    # for city/table-related queries
    if any(word in user_query for word in city_keywords):
        return "The available cities in the database are Pune, Solapur, Chennai, Erode, Jabalpur, Thanjavur, and Tiruchirappalli."

    question_keywords = {"possible", "questions", "ask", "database", "type"}      # for possible question-related queries
    if any(word in user_query for word in question_keywords):
        return """
        The possible questions you can ask are:
        - What was the total property tax collection in 2013-14 residential for Aundh in Pune city?
        
        - What was the property efficiency for the year 2015-16 commercial for Chennai?
        - What was the collection gap for the year 2016-17 residential for Thanjavur?
        - What was the collection gap for Solapur from 2013-18 residential?
        - What will be the tax demand for the year 2025 in Pune for residential?
        - What will be the property efficiency (residential) for the year 2019 in Pune?
        """
    return None


if user_query and user_query.strip():  # process user query
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    response = handle_edge_cases(user_query)   # check for edge cases
    
    if not response:
        city, property_type, year = extract_query_info(user_query)  # extract query info
        df = None  # load the dataset
        if city:
            df_path = f"D:/TaxQueryAI/datasets/transformed_data/Property-Tax-{city}.csv"
            if os.path.exists(df_path):
                df = pd.read_csv(df_path)
        if df is not None:  # generate response
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history, city, property_type, year, df)
        else:
            response = "Sorry, I couldn't find anything related to that. Please check your input."

    if response:  # append AI response and display it
        st.session_state.chat_history.append(AIMessage(content=response))
        with st.chat_message("AI"):
            st.markdown(response)

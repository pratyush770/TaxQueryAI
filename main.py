import os
from secret_key import sec_key
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from functions import get_response

os.environ['GROQ_API_KEY'] = sec_key  # set environment variable

if "db" not in st.session_state:  # initialize database
    mysql_uri = "mysql+mysqlconnector://root_readonly:matsumoto@localhost:3307/property_tax"
    st.session_state.db = SQLDatabase.from_uri(mysql_uri)

if "chat_history" not in st.session_state:  # initialize chat_history
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm a SQL assistant, Ask me anything about your database")
    ]


st.set_page_config(  # page configuration
    page_title="Chat with MySQL",
    page_icon=":speech_balloon:"
)

st.title("MySQL Database Q&A Tool")  # gives title
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):  # for AIMessage
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):  # for HumanMessage
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")  # asks user for prompt
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))  # update chat_history
    with st.chat_message("Human"):
        st.markdown(user_query)  # display human message

    with st.chat_message("AI"):
        # sql_chain = get_sql_chain(st.session_state.db)  # for query testing
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        # response = sql_chain.invoke({  # for query testing
        #     "chat_history": st.session_state.chat_history,
        #     "question": user_query
        # })
        st.markdown(response)  # display ai message
    st.session_state.chat_history.append(AIMessage(content=response))

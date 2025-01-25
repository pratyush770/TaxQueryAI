import os
from secret_key import sec_key
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser  # for parsing the output into a string
from langchain_core.runnables import RunnablePassthrough  # allows the function to pass as a runnable
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

os.environ['HUGGINGFACEHUB_API_TOKEN'] = sec_key  # set environment variable
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

model_name = "llama-3.1-8b-instant"  # name of model used
llm = ChatGroq(
    model_name=model_name,
    temperature=0.1,  # more accurate results
    groq_api_key=sec_key
)

if "db" not in st.session_state:
    mysql_uri = "mysql+mysqlconnector://root_readonly:matsumoto@localhost:3307/property_tax"
    st.session_state.db = SQLDatabase.from_uri(mysql_uri)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm a SQL assistant, Ask me anything about your database")
    ]


def get_sql_chain(db: SQLDatabase):
    template = """
        Based on the table schema below, write only a SQL query that would answer the user's question:
        {schema}
        Conversation History: {chat_history}

        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

        For example:
        Question: which 3 artists have the most tracks?
        SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
        Question: Name 10 artists
        SQL Query: SELECT Name FROM Artist LIMIT 10;

        Your turn:

        Question: {question}
        SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    template = """
       You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
       Based on the table schema below, question, SQL query, and SQL response, write only a natural language response to the user's question.
       <SCHEMA>{schema}</SCHEMA>

       Conversation History: {chat_history}
       SQL Query: <SQL>{query}</SQL>
       User question: {question}
       SQL Response: {response}"""
    prompt = ChatPromptTemplate.from_template(template)  # another template to get the natural language response

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
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


st.set_page_config(
    page_title="Chat with MySQL",
    page_icon=":speech_balloon:"
)

st.title("MySQL Database Q&A Tool")
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # sql_chain = get_sql_chain(st.session_state.db)
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))

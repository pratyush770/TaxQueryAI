from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser  # for parsing the output into a string
from langchain_core.runnables import RunnablePassthrough  # allows the function to pass as a runnable
from langchain_groq import ChatGroq
from secret_key import sec_key

model_name = "llama-3.1-8b-instant"  # name of model used
llm = ChatGroq(
    model_name=model_name,
    temperature=0.1,  # more accurate results
    groq_api_key=sec_key
)

SCHEMA_CACHE = None  # Global schema cache


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
        Question: What was the total property tax collection in 2013-14 residential for aundh in pune city?
        SQL Query: SELECT SUM(Tax_Collection_Cr_2013_14_Residential) AS total_tax_collected FROM pune WHERE Ward_Name = "Aundh";
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


def get_response(user_query: str, db: SQLDatabase, chat_history: list):  # function to display natural language response output
    # Check if the user is expressing gratitude
    gratitude_keywords = ["thanks", "thank you", "thx", "appreciate", "grateful"]
    if any(word in user_query.lower() for word in gratitude_keywords):
        return "You're welcome! Let me know if you need anything else."
    sql_chain = get_sql_chain(db)
    template = """
       Based on the table schema below, question, SQL query, and SQL response, write only a natural language response to the user's question.
       {schema}
       Conversation History: {chat_history}
       
       Question: {question}
       SQL Query: {query}
       SQL Response: {response}
       Natural Language Response:
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

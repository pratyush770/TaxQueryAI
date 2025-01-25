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


def get_sql_chain(db: SQLDatabase):  # function to get sql query
    template = """
        Based on the table schema below, write only a SQL query that would answer the user's question.
        Don't provide any extra information other than the sql query.
        {schema}
        Conversation History: {chat_history}

        Question: {question}
        SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):  # function to get schema info
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):  # function to display natural language response output
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

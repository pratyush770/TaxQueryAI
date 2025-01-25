from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser  # for parsing the output into a string
from langchain_core.runnables import RunnablePassthrough  # allows the function to pass as a runnable


def get_sql_chain(db: SQLDatabase, llm):
    template = """
        Based on the table schema below, write only a SQL query that would answer the user's question:
        {schema}
        Conversation History: {chat_history}

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

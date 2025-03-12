from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else.
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def validate_query(sql_query: str) -> bool:
    """
    Prevents execution of UPDATE and DELETE queries.
    """
    sql_query_lower = sql_query.lower()
    if "update" in sql_query_lower or "delete" in sql_query_lower:
        return False
    return True

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    chat_history = chat_history[-5:]  # Pass only the last 5 chat messages
    
    general_messages = {"hello": "Hello! How can I assist you today?", 
                        "hi": "Hi there! How can I help?", 
                        "how are you": "I'm just a bot, but I'm here to help!", 
                        "bye": "Goodbye! Have a great day!", 
                        "goodbye": "Take care! See you next time!", 
                        "have a good day": "You too! Let me know if you need anything."}
    
    if user_query.lower() in general_messages:
        return "", general_messages[user_query.lower()]
    
    sql_chain = get_sql_chain(db)
    sql_query = sql_chain.invoke({"question": user_query, "chat_history": chat_history})
    
    if not validate_query(sql_query):
        return "", "Update and Delete operations are not allowed. Please ask a read-only query."
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    
    chain = (
        RunnablePassthrough.assign(
            schema=lambda _: db.get_table_info(),
            response=lambda _: db.run(sql_query),
            query=lambda _: sql_query
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "query": sql_query
    })
    
    return sql_query, response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title("Chat with MySQL")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", value="Chinook", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

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
        sql_query, response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        
        if sql_query:
            with st.expander("Show Generated SQL Query"):
                st.code(sql_query, language="sql")
        
        st.markdown("**Response:**")
        st.markdown(response)
    
    st.session_state.chat_history.append(AIMessage(content=response))

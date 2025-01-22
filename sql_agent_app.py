import streamlit as st
import os
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import sqlite3
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv()

# Initialize LangSmith client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sql-agent-app"  # Your project name

# If you have a LangSmith API key, uncomment and add it to your .env file
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Function to initialize the Chinook database
def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database and agent if not already done
if "agent_executor" not in st.session_state:
    # Initialize database and agent
    engine = get_engine_for_chinook_db()
    db = SQLDatabase(engine)
    
    MODEL = "command-r-plus"
    llm = ChatCohere(
        model=MODEL, 
        temperature=0.1,
        verbose=True,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    context = toolkit.get_context()
    
    # Store database information in session state
    st.session_state.table_names = context.get('table_names', '')
    st.session_state.table_info = context.get('table_info', '')
    st.session_state.context = context
    
    tools = toolkit.get_tools()
    
    # Create the preamble with context
    preamble = f"""## Task And Context
You use your advanced complex reasoning capabilities to help people by answering their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You may need to use multiple tools in parallel or sequentially to complete your task. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

## Additional Information
You are an expert who answers the user's question by creating SQL queries and executing them.
You are equipped with a number of relevant SQL tools.

Here is information about the database:
{st.session_state.table_info}

Question: {{input}}"""
    
    # Create the agent with the preamble
    prompt = ChatPromptTemplate.from_template(preamble)
    agent = create_cohere_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

# Streamlit UI
st.title("SQL Agent Chat Interface")

# Add a sidebar with database information
with st.sidebar:
    st.header("Database Information")
    with st.expander("Available Tables"):
        st.write(st.session_state.table_names)
    with st.expander("Database Schema"):
        st.code(st.session_state.table_info)

st.write("Ask questions about the Chinook music database!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the database"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent_executor.invoke({
                "input": prompt,
                "table_info": st.session_state.context
            })
            response_content = response["output"]
            st.markdown(response_content)
            
            # Show intermediate steps in an expander
            with st.expander("See agent's thought process"):
                for step in response["intermediate_steps"]:
                    st.write(f"Tool: {step[0].tool}")
                    st.write(f"Input: {step[0].tool_input}")
                    st.write(f"Output: {step[1]}")
                    st.write("---")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
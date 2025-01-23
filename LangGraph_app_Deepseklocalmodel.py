import os
import uuid
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI  # Updated import
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
import chainlit as cl
from dotenv import load_dotenv
#import openai


# Load environment variables
load_dotenv()

# Access API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize memory for checkpointing
memory = MemorySaver()

# Define the state of the graph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Initialize the graph builder
graph_builder = StateGraph(State)

# Initialize the Tavily search tool
tool = TavilySearchResults(max_results=5)
tools = [tool]

# Initialize the local LM Studio model
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # LM Studio server URL
    api_key="not-needed",  # Dummy API key (not required for LM Studio)
    model="local-model",  # Generic name for local model
    temperature=0.7,
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add the tools node to the graph
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Add conditional edges to decide whether to use tools
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Add edges to connect the nodes
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph with memory checkpointing
graph = graph_builder.compile(checkpointer=memory)

# Chainlit event handlers
@cl.on_chat_start
async def start():
    # Generate a unique thread ID for the session
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    
    # Configure the checkpoint
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = memory.get(config)
    
    # Initialize messages in the session
    if checkpoint and "messages" in checkpoint["state"]:
        cl.user_session.set("messages", checkpoint["state"]["messages"])
    else:
        cl.user_session.set("messages", [])
    
    # Store the graph in the session
    cl.user_session.set("graph", graph)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the graph and thread ID from the session
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}
    
    # Retrieve existing messages from the session
    existing_messages = cl.user_session.get("messages")
    
    # Append the new user message
    user_message = HumanMessage(content=message.content)
    existing_messages.append(user_message)
    
    # Invoke the graph with the updated messages
    events = await graph.ainvoke(
        {"messages": existing_messages},
        config,
    )
    
    if "messages" in events:
        # Extract the assistant's response
        response = events["messages"][-1].content
        
        # Append the assistant's response to the session messages
        cl.user_session.get("messages").append(AIMessage(content=response))
        
        # Send the response to the user
        await cl.Message(content=response).send()
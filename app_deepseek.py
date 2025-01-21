import os
import uuid
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
import chainlit as cl
from dotenv import load_dotenv
from openai import OpenAI  # Use the OpenAI client for DeepSeek

# Load environment variables
load_dotenv()

# Access API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Add DeepSeek API key to .env

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

# Initialize the DeepSeek client
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",  # DeepSeek API endpoint
)

# Define the chatbot node
def chatbot(state: State):
    try:
        # Convert LangChain messages to DeepSeek API format
        messages = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in state["messages"]
        ]
        
        # Call the DeepSeek API
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek model name
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )
        
        # Extract the assistant's response
        assistant_message = response.choices[0].message.content
        return {"messages": [AIMessage(content=assistant_message)]}
    except Exception as e:
        print(f"Error invoking DeepSeek API: {e}")
        return {"messages": [AIMessage(content="Sorry, I encountered an error. Please try again.")]}

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
    try:
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
    except Exception as e:
        print(f"Error during graph invocation: {e}")
        await cl.Message(content="Sorry, I encountered an error. Please try again.").send()
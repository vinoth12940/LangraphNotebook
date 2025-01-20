import os
import uuid

from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import chainlit as cl
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=5)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)

@cl.on_chat_start
async def start():
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = memory.get(config)
    if checkpoint and "messages" in checkpoint:
        cl.user_session.set("messages", checkpoint["messages"])
    else:
        cl.user_session.set("messages", [])
    cl.user_session.set("graph", graph)

@cl.on_message
async def main(message: cl.Message):
    graph = cl.user_session.get("graph")
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
    
    # Retrieve the existing messages from the session
    existing_messages = cl.user_session.get("messages")
    
    # Append the new user message
    existing_messages.append(HumanMessage(content=message.content))
    
    # Invoke the graph with the updated messages
    events = await graph.ainvoke(
        {"messages": existing_messages},
        config,
    )
    
    if "messages" in events:
        response = events["messages"][-1].content
        # Append the assistant's response to the session messages
        cl.user_session.get("messages").append(AIMessage(content=response))
        await cl.Message(content=response).send()
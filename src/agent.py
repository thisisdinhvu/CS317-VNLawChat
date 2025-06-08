import os
from langchain.chat_models import init_chat_model
from typing import Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from src.retriever_tool import retrieve_from_pinecone


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY in environment.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def add (a : int, b : int) -> int:
    """Adds two integers and returns the result."""
    print(f"Tool 'add' called with inputs: a={a}, b={b}")
    return a + b
def subtract (a : int, b : int) -> int:
    """Subtract two numbers and returns the result."""
    print(f"Tool 'substract' called with inputs: a={a}, b={b}")
    return a - b

tools = [add, subtract, retrieve_from_pinecone]
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    """
    Processes the conversation state and generates a response.
    If a tool has been called, uses the tool's result to answer the user's question.
    Otherwise, lets the LLM decide to answer directly or call a tool.
    """
    messages = state["messages"]
    user_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    if not user_message:
        return {"messages": [AIMessage(content="Error: No user message found.")]}

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if msg.name in ["add", "subtract"]:
                return {"messages": [AIMessage(content=f"The result is {msg.content}.")]}
            elif msg.name == "retrieve_from_pinecone":
                prompt = (
                f"You are a helpful assistant. The user asked: '{user_message.content}'.\n\n"
                "Here are some retrieved documents from a Vietnamese law database:\n"
                f"{msg.content}\n\n"
                "First, determine if these documents contain information relevant to the user's question. "
                "If they do, provide a concise and accurate answer to the user's question based on the documents. "
                "Format the answer in a user-friendly way, using bullet points or a short paragraph as appropriate. "
                "If the documents are not relevant to the question, respond with: 'I don’t have any information related to the question.'"
                "Everything you say must be in Vietnamese"
                )
                response = llm.invoke([HumanMessage(content=prompt)])
                return {"messages": [response]}

    system_message = SystemMessage(
        content="You are a helpful assistant. You can answer any question to the best of your knowledge, using the available tools (add, subtract, retrieve_from_pinecone) only when necessary.")
    response = llm_with_tools.invoke([system_message, user_message])
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def run_agent(user_input: str) -> str:
    state = {"messages": [{"role": "user", "content": user_input}]}
    response = None
    for event in graph.stream(state):
        for value in event.values():
            last_msg = value["messages"][-1]
            if isinstance(last_msg, AIMessage):
                response = last_msg.content
    return response or "Không có phản hồi từ mô hình."

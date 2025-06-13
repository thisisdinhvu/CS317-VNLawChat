import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from src.retriever_tool import retrieve_from_pinecone
from src.add_memory_tool import ChatMemory, create_retrieve_memory_tool, create_add_memory_tool
from dotenv import load_dotenv
load_dotenv()



GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINCONE_CHAT_HISTORY_INDEX = os.getenv('PINCONE_CHAT_HISTORY_INDEX')


def add (a : int, b : int) -> int:
    """Adds two integers and returns the result."""
    print(f"Tool 'add' called with inputs: a={a}, b={b}")
    return a + b
def subtract (a : int, b : int) -> int:
    """Subtract two numbers and returns the result."""
    print(f"Tool 'substract' called with inputs: a={a}, b={b}")
    return a - b

chat_mem = ChatMemory(
    pinecone_api_key=PINECONE_API_KEY,
    index_name=PINCONE_CHAT_HISTORY_INDEX
)

retrieve_memory_tool = create_retrieve_memory_tool(chat_mem)
add_memory_tool = create_add_memory_tool(chat_mem)

tools = [add,subtract, retrieve_from_pinecone,retrieve_memory_tool, add_memory_tool]
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


    # Check for the latest user message
    user_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

    # print(user_message)
    if not user_message:
        return {"messages": [AIMessage(content="Error: No user message found.")]}

    # Check for a ToolMessage (result of a tool call)

    tool_response = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)
    # print(tool_response)

    if tool_response:
        # print('here in tool')
        if tool_response.name in ["add", "subtract"]:
            # For arithmetic tools, return the result directly
            return {"messages": [AIMessage(content=f"The result is {tool_response.content}.")]}
        elif tool_response.name == "retrieve_from_pinecone":
            # print(tool_response.content)

            # Construct a prompt to determine if documents are relevant and to format the answer
            prompt = (
                f"You are a helpful assistant. The user asked: '{user_message.content}'.\n\n"
                "Here are some retrieved documents from a Vietnamese law database:\n"
                f"{tool_response.content}\n\n"
                "First, determine if these documents contain information relevant to the user's question. "
                "If they do, provide a concise and accurate answer to the user's question based on the documents. "
                "Format the answer in a user-friendly way, using bullet points or a short paragraph as appropriate. "
                "If the documents are not relevant to the question, respond with: 'I don’t have any information related to the question.'"
                "Everything you say must be in Vietnamese"
            )
            response = llm.invoke([HumanMessage(content=prompt)])

            try:
                chat_mem.add_memory(user_message.content, response.content)
            except Exception as e:
                print(f"Error storing memory after legal Q&A: {e}")
            return {"messages": [response]}

        elif tool_response.name == "retrieve_memory":
            # print(tool_response.content)
            prompt = (
                f"You are a helpful assistant. The user asked: '{user_message.content}'.\n\n"
                "Here is the relevant past conversation context:\n"
                f"{tool_response.content}\n\n"
                "Use this context to generate an accurate response."
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            return {"messages": [response]}
        elif tool_response.name == "add_memory":
            # Simply acknowledge once and stop further processing for this turn
            return {"messages": [AIMessage(content=f"{tool_response.content}")]}

    # No tool response, let the LLM decide what to do with a clear system instruction
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant with access to the following tools:\n"
            "- add(a: int, b: int): Adds two integers.\n"
            "- subtract(a: int, b: int): Subtracts two integers.\n"
            "- retrieve_from_pinecone(query_text: str): Retrieves legal documents for Vietnamese law questions.\n"
            "- retrieve_memory(query: str, top_k: int): Retrieves past conversation context. Use this BEFORE responding if the user asks about prior conversation, e.g., 'What did I say earlier?', 'What’s my name?'.\n"
            "- add_memory(user_message: str, ai_message: str): Stores the user-AI message pair. Call this ONLY if the exchange contains user-specific information (e.g., user's name, preferences) or questions about Vietnamese laws.\n\n"
            "For each user query:\n"
            "1. If the query references past conversation (e.g., 'What did I ask?', 'What’s my name?'), call retrieve_memory with the query.\n"
            "2. Generate a response, using retrieved memory or legal documents if applicable.\n"
            "3. Decide if the exchange should be stored (e.g., contains user's name or legal questions). If so, call add_memory with the user's message and your response.\n"
            "For legal questions about Vietnamese laws, use retrieve_from_pinecone. For arithmetic, use add or subtract."
        )
    )
    # print('here out tool')
    response = llm_with_tools.invoke([system_message, user_message.content])

    return {"messages": [response]}


# Build the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def run_agent(user_input: str, thread_id: str) -> str:
    """
    Chạy LangGraph agent với input từ người dùng và thread_id.
    Trả về nội dung phản hồi cuối cùng từ AI dưới dạng chuỗi.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"messages": [HumanMessage(content=user_input)]}

    final_response = ""

    for event in graph.stream(initial_state, config):
        for value in event.values():
            last_message = value["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                final_response = last_message.content

    return final_response or "Không có phản hồi từ mô hình."

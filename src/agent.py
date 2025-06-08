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

def add(a: int, b: int) -> int:
    """Cộng hai số nguyên a và b."""
    print(f"Tool 'add' called with inputs: a={a}, b={b}")
    return a + b

def subtract(a: int, b: int) -> int:
    """Trừ số nguyên b khỏi a."""
    print(f"Tool 'subtract' called with inputs: a={a}, b={b}")
    return a - b

tools = [add, subtract, retrieve_from_pinecone]
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
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
                    f"Bạn là trợ lý luật. Người dùng hỏi: '{user_message.content}'.\n\n"
                    f"Các tài liệu truy xuất:\n{msg.content}\n\n"
                    "Hãy xác định các tài liệu có liên quan không. Nếu có, trả lời ngắn gọn, rõ ràng bằng tiếng Việt. "
                    "Nếu không liên quan, trả lời: 'Tôi không có thông tin liên quan đến câu hỏi này.'"
                )
                response = llm.invoke([HumanMessage(content=prompt)])
                return {"messages": [response]}

    system_message = SystemMessage(
        content="Bạn là trợ lý luật Việt Nam. Sử dụng công cụ (add, subtract, retrieve_from_pinecone) nếu cần.")
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

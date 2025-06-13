# simple_chat_memory.py

import os
import uuid
from typing import Optional, List, Dict
from pinecone import Pinecone
from google import genai
from langchain_core.tools import tool
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Configure Gemini API client once
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_embedding(text: str) -> Optional[List[float]]:
    if not text:
        print("Error: Empty text provided for embedding")
        return None
    try:
        result = genai_client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text
        )
        embeddings = result.embeddings
        # Check if embeddings is a list with one element
        if isinstance(embeddings, list) and len(embeddings) == 1:
            embedding_vector = embeddings[0].values  # Extract the actual vector
            # Verify it's a list of floats
            if isinstance(embedding_vector, list) and all(isinstance(x, float) for x in embedding_vector):
                # print(f"Generated embedding vector of length: {len(embedding_vector)}")
                return np.array(embedding_vector)
            else:
                print("Unexpected embedding format: not a list of floats")
                return None
        else:
            print("Unexpected embeddings structure: not a list with one element")
            return None
    except Exception as e:
        print(f"Error obtaining Gemini embedding: {e}")
        return None

class ChatMemory:
    def __init__(self, pinecone_api_key: Optional[str] = None, index_name: Optional[str] = None, namespace: str = "chat-memory"):
        api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        idx_name = index_name or os.getenv("PINECONE_INDEX")
        if not api_key or not idx_name:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(idx_name)
        self.namespace = namespace
        print(f"Initialized ChatMemory with index: {idx_name}, namespace: {namespace}")

    def retrieve_memory(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve past conversation context based on the query."""
        # print(f"Attempting to retrieve memory for query: {query}")
        embeddings = get_gemini_embedding(query)
        if embeddings is None:
            print("Embedding failure on retrieval query.")
            return []
        try:
            query_response = self.index.query(
                vector=embeddings.tolist(),
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            matches = query_response.get("matches", [])
            # print(f"Retrieved {len(matches)} matches from Pinecone")
            return [{"user_message": m["metadata"]["user_message"], "ai_message": m["metadata"]["ai_message"]} for m in matches]
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def add_memory(self, user_message: str, ai_message: str) -> None:
        combined_text = f"User: {user_message}\nAI: {ai_message}"
        embeddings = get_gemini_embedding(combined_text)
        if embeddings is None:
            print("Skipping upsert due to embedding failure.")
            return
        try:
            unique_id = str(uuid.uuid4())
            self.index.upsert(
                vectors=[{
                    "id": unique_id,
                    "values": embeddings,
                    "metadata": {"user_message": user_message, "ai_message": ai_message}
                }],
                namespace=self.namespace
            )
            # print(f"Stored memory with ID: {unique_id}")
        except Exception as e:
            print(f"Upsert error: {e}")

# Factory functions to create tools
def create_retrieve_memory_tool(chat_mem: 'ChatMemory'):
    @tool
    def retrieve_memory(query: str, top_k: int = 3) -> str:
        """
        Retrieve past conversation context based on the user's query.
        Use this tool when you need to recall previous exchanges to answer the user.
        """
        try:
            print('using retrieve_memory tool')
            memory_results = chat_mem.retrieve_memory(query, top_k)
            if not memory_results:
                print("No relevant past conversation found.")
                return "No relevant past conversation found."
            formatted_results = "\n".join([f"User: {item['user_message']}\nAI: {item['ai_message']}" for item in memory_results])
            # print(f"Formatted memory results: {formatted_results[:100]}...")
            return formatted_results
        except Exception as e:
            print(f"Error retrieving memory: {str(e)}")
            return f"Error retrieving memory: {str(e)}"
    return retrieve_memory

def create_add_memory_tool(chat_mem: 'ChatMemory'):
    @tool
    def add_memory(user_message: str, ai_message: str) -> str:
        """
        Store the current user-AI message pair for future reference.
        Use this tool only to store user's personal information, user's questions about Vietnamese laws.
        """
        try:
            print('using add_memory tool')
            chat_mem.add_memory(user_message, ai_message)
            return "Memory stored successfully."
        except Exception as e:
            print(f"Error storing memory: {str(e)}")
            return f"Error storing memory: {str(e)}"
    return add_memory
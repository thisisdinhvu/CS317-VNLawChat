from gradio_client import Client
import json
from pinecone import Pinecone
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

# Pinecone setup (replace with your API key and index name)
PINECONE_API = os.getenv('PINECONE_API_KEY')
INDEX_NAME = os.getenv('PINCONE_LEGAL_DOCUMENT_INDEX')
NAMESPACE = "bkai-legal-docs"

pc = Pinecone(api_key=PINECONE_API)


def get_index(index_name: str):
    """
    Create the Pinecone index if it doesn't exist, then return the Index object.
    """
    return pc.Index(index_name)


# Initialize Pinecone index
index = get_index(INDEX_NAME)


def get_embeddings(text: str) -> list:
    """
    Sends text to the Hugging Face Space API and returns a 768-dimensional embedding vector.

    Args:
        text (str): The input text to be embedded.

    Returns:
        list: A list of 768 float values representing the embeddings.

    Raises:
        ValueError: If the input text is empty, or the API response is invalid.
        Exception: For other API-related errors.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    try:

        client = Client("ICTuniverse/testing-n8n", verbose=False)
        result = client.predict(
            text=text,
            api_name="/predict",
        )
        if isinstance(result, str):
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                embeddings = [float(parsed[str(i)]) for i in range(768)]
            else:
                raise ValueError("Expected a JSON dictionary with indices 0 to 767")
        elif isinstance(result, list):
            embeddings = result
        else:
            raise ValueError("API response must be a list or a JSON string")

        if len(embeddings) != 768:
            raise ValueError(f"Expected 768-dimensional embeddings, got {len(embeddings)}")
        if not all(isinstance(x, (int, float)) for x in embeddings):
            raise ValueError("Embeddings must contain only numeric values")

        return embeddings
    except Exception as e:
        raise Exception(f"Error fetching embeddings from API: {str(e)}")


@tool
def retrieve_from_pinecone(query_text: str) -> str:
    """
    Retrieve related contents to user's query, specifically about Vietnamese legal questions.
    Use this tool for questions about Vietnamese laws, regulations, or legal cases.

    Args:
        query_text (str): The input query text to search for.

    Returns:
        str: A formatted string containing the related content from the top 3 matches.
    """
    if not query_text or not query_text.strip():
        return "Error: Query text cannot be empty"

    try:
        print('using retriever tool')
        embeddings = get_embeddings(query_text)
        query_response = index.query(
            vector=embeddings,
            top_k=5,
            include_metadata=True,
            namespace=NAMESPACE
        )
        matches = query_response.get("matches", [])
        if not matches:
            return "No matching documents found."

        results = []
        for i, match in enumerate(matches, 1):
            metadata = match.get("metadata", {})
            if "text" in metadata:
                results.append(f"{i}. {metadata['text']}")
            else:
                results.append(f"{i}. No content available for this document.")
        print(results)
        return "\n".join(results)
    except Exception as e:
        return f"Error retrieving from Pinecone: {str(e)}"
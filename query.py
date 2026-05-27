from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

load_dotenv()

client = OpenAI()

qdrant_client = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="hungry_collection",
    embedding=embeddings,
)

async def ask_query(user_query):

    results = vector_store.similarity_search(
        query=user_query,
        k=5
    )


    context = "\n\n".join([
        doc.page_content
        for doc in results
    ])
    # print("context", context)

    prompt = f"""
        You are a food recommendation assistant.

        User Query:
        {user_query}

        Retrieved Cafe Data:
        {context}

        I will ask questions about restaurants, cafes, food carts and places where I can go to eat. 
        The places will have attributes such as ambience, vibes, aesthetic, location, postive reviews, negative reviews, food items, service, who to go with, parking, etc.
        You have to answer in 2 lines only. Give answer from context only . If the answer is not in context, you can give a suggestion from your own.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role":"user",
                "content": prompt
            }
        ]
    )

    print("🤖", response.choices[0].message.content)
    print("Tokens used: ", response.usage.completion_tokens)
    return response.choices[0].message.content
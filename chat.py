from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import asyncio
from db_connection import connection
from prompts import SYSTEM_PROMPT, update_active_restaurant_prompt, extract_restaurant_data_prompt, generate_followup_question_prompt, generate_restaurant_summary_prompt, suggest_places_prompt, task_selector_prompt

load_dotenv()

client = OpenAI()
qdrant_client = QdrantClient(url="http://localhost:6333")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Create collection if missing
if not qdrant_client.collection_exists("hungry_collection"):

    qdrant_client.create_collection(
        collection_name="hungry_collection",
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        ),
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="hungry_collection",
    embedding=embeddings,
)

def save_vector_in_db(
    summary_text: str,
    metadata: dict
):
    document = Document(
        page_content=summary_text,
        metadata=metadata
    )

    vector_store.add_documents([document])

class RestaurantMemory(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None

    ambience: Optional[str] = None
    service: Optional[str] = None
    price_range: Optional[str] = None

    food_items: List[str] = []

    companions: List[str] = []

    positive_review: Optional[str] = None
    negative_review: Optional[str] = None

    other_information: Optional[str] = None

class RestaurantExtraction(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None

    ambience: Optional[str] = None
    service: Optional[str] = None
    price_range: Optional[str] = None

    food_items: Optional[list[str]] = None
    companions: Optional[list[str]] = None

    positive_review: Optional[str] = None
    negative_review: Optional[str] = None

    other_information: Optional[str] = None

async def extract_restaurant_data(
    current_memory: dict,
    user_message: str
):
    prompt = extract_restaurant_data_prompt(current_memory, user_message)

    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=RestaurantExtraction,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.parsed

def merge_memory(existing, new_data):

    new_data = new_data.model_dump(
        exclude_none=True
    )

    for key, value in new_data.items():

        if isinstance(value, list):

            existing.setdefault(key, [])

            for item in value:
                if item not in existing[key]:
                    existing[key].append(item)

        else:
            existing[key] = value

    return existing

REQUIRED_FIELDS = [
    "name",
    "location",
    "category",
    "ambience",
    "food_menu",
    "favourite_food",
    "service",
    "price_range",
    "positive_review",
    "negative_review",
    "additional_information"
]

def get_missing_fields(memory):

    missing = []

    for field in REQUIRED_FIELDS:

        if not memory.get(field):
            missing.append(field)

    return missing

async def generate_followup_question(
    memory,
    missing_fields
):
    prompt = generate_followup_question_prompt(memory, missing_fields)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def new_restaurant():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  
        messages=message_history,
        max_tokens=200
    )

    tokens_used = response.usage.prompt_tokens

    raw_result = (response.choices[0].message.content)
    message_history.append({ "role": "assistant", "content": raw_result })

    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "START":
        print("🔥", parsed_result.get("content"))

    if parsed_result.get("step") == "QUESTION":
        # print("🧠", parsed_result.get("content"))
        answer = input("🧑🏻 ")
        message_history.append({"role": "user", "content": answer})

    if parsed_result.get("step") == "OUTPUT":
        # print("🤖", parsed_result.get("content"))
        save_vector_in_db(parsed_result.get("content"))
        # print(f"Token usage: ", tokens_used)
        
async def generate_restaurant_summary(memory):
    prompt = generate_restaurant_summary_prompt(memory)

    response = client.chat.completions.create(

        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

PRONOUNS = [
    "it",
    "there",
    "they",
    "them",
    "this",
    "that",
    "he",
    "she",
    "him",
    "her"
]

def resolve_query(user_query, memory):

    lower_query = user_query.lower()

    has_pronoun = any(
        pronoun in lower_query.split()
        for pronoun in PRONOUNS
    )

    active_restaurant = memory.get("active_restaurant")
    active_food = memory.get("active_food")
    active_location = memory.get("active_location")
    active_topic = memory.get("active_topic")

    if has_pronoun and (active_restaurant or active_location or active_food or active_topic):

        replacements = {
            "there": f"at {active_restaurant}",
            "it": active_restaurant,
            "this place": active_restaurant
        }

        resolved_query = lower_query

        for old, new in replacements.items():
            resolved_query = resolved_query.replace(old, new)

        return resolved_query

    return user_query

def update_active_restaurant(ai_response, memory):

    prompt = update_active_restaurant_prompt(ai_response) 

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type":"json_object"},
        messages=[
            {
                "role":"user",
                "content": prompt
            }
        ]
    )

    data = json.loads(
        response.choices[0].message.content
    )

    if data.get("restaurant_name"):
        memory["active_restaurant"] = data["restaurant_name"]
    if data.get("food_item"):
        memory["active_food"] = data["food_item"]
    if data.get("location"):
        memory["active_location"] = data["location"]
    if data.get("topic"):
        memory["active_topic"] = data["topic"]

class SuggestResponse(BaseModel):
    end_conversation: bool = False
    response: str = ""

async def suggest_places(user_query, memory):

    # 1. Resolve pronouns
    resolved_query = resolve_query(
        user_query,
        memory
    )
    print("\nResolved_query",resolved_query)

    # 2. Vector search
    results = vector_store.similarity_search(
        query=resolved_query,
        k=5
    )

    context = "\n\n".join([
        doc.page_content
        for doc in results
    ])

    print("\nContext: ", context)

    prompt = suggest_places_prompt(user_query, resolved_query, memory, context)

    # 3. Generate AI response
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=SuggestResponse,
        messages = [
            {
                "role":"user",
                "content": prompt
            }
        ]
    )

    # 4. Extract AI response
    result = response.choices[0].message.parsed
    ai_response = result.response

    # 5. Update active restaurant memory
    update_active_restaurant(
        ai_response,
        memory
    )

    # 6. Save conversation in MYSQL DB
    cursor = connection.cursor()
    insert_query = """
        INSERT INTO ai_response (context, user_query, ai_response)
        VALUES (%s, %s, %s)
    """

    cursor.execute(insert_query, (context, user_query, ai_response))
    connection.commit()

    # 7. Save conversation history
    conversation = memory.get("conversation", [])
        
    if result.end_conversation:
        memory["end_conversation"] = True
    else:
        memory["end_conversation"] = False
        # if more than 5 conversation, remove first element then add one at last
        if len(conversation) > 5: 
            conversation.pop(0)
        # add current query and response to conversation memory
        conversation.append({
            "user_query": user_query,
            "ai_response": ai_response
        })
        
    memory["conversation"] = conversation

    return {"memory": memory, "ai_response": ai_response}

class TaskSelectorOutput(BaseModel):
    tool: Literal[
        "new_restaurant",
        "suggest_places",
        "greeting"
    ]

    input: Optional[str] = None
    content: Optional[str] = None

async def task_selector(query):
    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening"
    ]

    if query.lower().strip() in greetings:
        return TaskSelectorOutput(
            tool="greeting",
            content="Hello! How can I help?"
        )

    SYSTEM_PROMPT = task_selector_prompt()

    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=TaskSelectorOutput,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    result = response.choices[0].message.parsed
    print("Task Selected ", result)
    return result


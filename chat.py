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
    prompt = f"""
    Current Restaurant Data:

    {json.dumps(current_memory, indent=2)}

    User Message:

    {user_message}

    Extract any restaurant information
    mentioned by the user.

    Return only JSON.
    """

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
    "food_items",
    "service",
    "price_range",
    "positive_review",
    "negative_review"
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
    prompt = f"""
    Known restaurant information:

    {json.dumps(memory, indent=2)}

    Missing fields:

    {missing_fields}

    Ask ONE natural conversational
    question to collect more information.
    """

    prompt = f"""
    Known restaurant information:

    {json.dumps(memory, indent=2)}

    Missing fields:

    {missing_fields}

    Ask ONE natural conversational
    question to collect more information.
    """

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

SYSTEM_PROMPT = """
    You are my personal AI Agent who remembers my food choices, cafes, restaurents ,food carts and places. 
    Your  have 2 tasks :-
    1. When I tell you about any food place, you have to ask specific questions in sequence, then return the response in a given structure.
    2. When I ask you for suggestions about food or places, you have to extract places and food from the previously stored responses and give me suggestions based on that.

    -> Task 1 : When I tell you about any food place, you have to ask specific questions in sequence, then return the response in a given structure.

    ### Step 1 : Ask questions and give response in specific structure
    ## Ask the following questions in sequence -
    # 1. What is the name, location and category of place?
    # 2. How was the ambience?
    # 3. What did you eat and what did you like and not like?
    # 4. How was the service?
    # 5. How was the price?
    # 6. Who did you go with?
    # 7. Positive review and negative review.
    # 8. Any other things you want to mention?
    This questions are based on the basic attributes. You can rephrase them based on user input. 

    ### Step 2: After asking all the questions, Generate the final response at the end in the following format.
    Cafe Name: Cafe Junoon

    Location: Smriti Nagar, Bhilai

    Category: Cafe

    Price Range: Medium

    Ambience:
    Cafe Junoon has aesthetic interiors with warm lighting and cozy seating. 
    The atmosphere is peaceful during weekdays and suitable for work meetings, dates, and casual hangouts.

    Popular Food:
    The cafe is known for white sauce pasta, cold coffee, and chocolate waffles.
    Customers especially enjoy the creamy pasta and thick cold coffee.

    Positive Reviews:
    Customers appreciate the polite staff, peaceful atmosphere, aesthetic decor, and fast service.

    Negative Reviews:
    Customers commonly complain about limited parking availability and weekend crowding.

    Best For:
    Cafe Junoon is ideal for couples, students, and remote workers.

    Other Information:
    Parking can sometimes be difficult during peak hours.

    Rules:  
    - Strictly follow the given questions and response format only.
    - Only ask one question at a time.
    - Understand the answer of each question. If the user wants to go to the next question, ask the next question. 
    
    Output JSON Format:
    {"step": "START" | "QUESTION" | "ANSWER" | "OUTPUT", "content": "string" }

    Example :- 
    {"step": "START", "content": "Well let's know more about the place"}
    {"step": "QUESTION", "content": "What is the name, location and category of place?"}
    {"step": "ANSWER", "content": "The place was Nukkad Cafe. It is in Nehru Nagar and it is a cafe."}
    {"step": "QUESTION", "content": "How was the ambience?"}
    {"step": "ANSWER", "content": "Ambience was good. The cafe was aeasthetic."}
    {"step": "QUESTION", "content": "What did you eat and what did you like and not like?"}
    {"step": "ANSWER", "content": "Let's skip to next question."}
    {"step": "QUESTION", "content": "How was the service?"}
    {"step": "ANSWER", "content": "Service was good."}
    {"step": "OUTPUT", "content": "Output in defined format."}

    -> Task 2: When I ask you for suggestions about food or places, you have to extract places and food from the previously stored responses and give me suggestions based on that.



"""

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
    prompt = f"""
    Restaurant Information:

    {json.dumps(memory, indent=2)}

    Create a restaurant summary in the following format:

    Cafe Name:
    Location:
    Category:
    Price Range:

    Ambience:

    Popular Food:

    Positive Reviews:

    Negative Reviews:

    Best For:

    Other Information:

    Keep it concise and natural.
    """

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

    prompt = f"""
    Extract restaurant name, food item, location of restaurant and the topic being talked about from text.

    Text:
    {ai_response}

    Return only JSON:
    {{
      "restaurant_name": string | null,
      "food_item": string | null,
      "location": string | null,
      "topic": string | null
    }}
    """

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

async def suggest_places(user_query, memory, query_number):

    # 1. Resolve pronouns
    resolved_query = resolve_query(
        user_query,
        memory
    )

    # 2. Vector search
    results = vector_store.similarity_search(
        query=resolved_query,
        k=5
    )

    context = "\n\n".join([
        doc.page_content
        for doc in results
    ])

    prompt = f"""
        You are a food recommendation assistant.

        Current User Query:
        {user_query}

        Resolved Query:
        {resolved_query}

        Conversation History:
        {memory.get("conversation", [])}

        Active Conversation Memory:

        * Current Restaurant: {memory.get("active_restaurant")}
        * Current Food Item: {memory.get("active_food")}
        * Current Location: {memory.get("active_location")}
        * Current Topic: {memory.get("active_topic")}

        Retrieved Restaurant Data:
        {context}

        The user may refer to previously discussed restaurants, cafes, places, or food items using pronouns such as:

        * it
        * there
        * they
        * them
        * this place
        * that cafe
        * he
        * she

        Always resolve these references using Active Conversation Memory and Conversation History.

        Important Rules:

        * Never invent unrelated restaurants.
        * If the user says "there", "it", or similar pronouns, assume they refer to the latest relevant restaurant unless context clearly changes.
        * Prefer Retrieved Restaurant Data first.
        * Use Conversation History when retrieval is insufficient.
        * If information is unavailable, give a reasonable suggestion.

        Return ONLY valid JSON in this format:

        {{
            "end_conversation": boolean,
            "response": string
        }}

        Additional Rules:

        * Response must be concise.
        * Maximum 2 lines.
        * No markdown.
        * No explanations outside JSON.
        * If user says thanks/bye/etc, set end_conversation=true and response="".
    """

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
        if query_number > 5: 
            conversation.pop(0)
        # add current query and response to conversation memory
        conversation.append({
            "user_query": user_query,
            "ai_response": ai_response
        })
        memory["query_count"] = query_number
        
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

    SYSTEM_PROMPT = """
        You are a routing assistant.

        Available tools:

        1. new_restaurant
        - Use when user shares details about a restaurant,
            cafe, food truck, food cart, or food experience.

        2. suggest_places
        - Use when user asks for recommendations,
            suggestions, or where to eat.

        3. greeting
        - Use for greetings or unrelated messages.

        Return only valid JSON matching the schema.
    """

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
    return result


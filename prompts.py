import json




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


def update_active_restaurant_prompt(ai_response) :

    return  f"""
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


def extract_restaurant_data_prompt(current_memory, user_message):
    
    return f"""
        Current Restaurant Data:

        {json.dumps(current_memory, indent=2)}

        User Message:

        {user_message}

        Extract any restaurant information
        mentioned by the user.

        Return only JSON.
    """ 


def generate_followup_question_prompt(memory, missing_fields):
     return f"""
    Known restaurant information:

    {json.dumps(memory, indent=2)}

    Missing fields:

    {missing_fields}

    Ask ONE natural conversational
    question to collect more information.
    """


def generate_restaurant_summary_prompt(memory):
    return f"""
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

    If user inputs extra information which does not fit in these topics, create a topic by your own and save the extra information, do not skip it. Keep it concise and natural.
    """


def suggest_places_prompt(user_query, resolved_query, memory, context):

    return f"""
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
        * Maximum 1 line.
        * No markdown.
        * No explanations outside JSON.
        * If user says thanks/bye/etc, set end_conversation=true and response="".
    """


def task_selector_prompt():
    return """
        You are a routing assistant.

        Available tools:

        1. new_restaurant
        - Use when user shares details about a restaurant,
            cafe, food truck, food cart, or food experience.

        2. suggest_places
        - Use when user asks for recommendations,
            suggestions, or where to eat.
        - when user asks for any information about any restaurant

        3. greeting
        - Use for greetings or unrelated messages.

        Return only valid JSON matching the schema.
    """


















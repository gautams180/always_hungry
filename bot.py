from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from chat import task_selector, new_restaurant, suggest_places, extract_restaurant_data, merge_memory, get_missing_fields, generate_followup_question, generate_restaurant_summary, save_vector_in_db
from telegram.ext import CommandHandler
from dotenv import load_dotenv
import os

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍔 Hey! I am your food memory AI agent."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_message = update.message.text

    if "restaurant_memory" in context.user_data:

        memory = context.user_data["restaurant_memory"]

        extracted = await extract_restaurant_data(
            memory,
            user_message
        )

        memory = merge_memory(
            memory,
            extracted
        )

        context.user_data["restaurant_memory"] = memory

        missing_fields = get_missing_fields(
            memory
        )

        if len(missing_fields) == 0:

            # Generate summary
            # Save to Qdrant
            # Clear session

            summary = await generate_restaurant_summary(
                memory
            )

            save_vector_in_db(
                summary_text=summary,
                metadata=memory
            )

            await update.message.reply_text(
                f"✅ Restaurant saved!\n\n{summary}"
            )

            del context.user_data["restaurant_memory"]

            return

        question = await generate_followup_question(
            memory,
            missing_fields
        )

        await update.message.reply_text(
            question
        )

        return

    if "conversation_memory" in context.user_data:

        memory = context.user_data["conversation_memory"]
        query_number = memory["query_count"] + 1

        response = await suggest_places(user_message, memory, query_number)

        updated_memory = response["memory"]
        context.user_data[
            "conversation_memory"
        ] = updated_memory

        if updated_memory.get("end_conversation"):
            await update.message.reply_text(
                f"✅ Conversation Ended!"
            )

            del context.user_data["conversation_memory"]

            return

        await update.message.reply_text(response["ai_response"])

        return

    task = await task_selector(user_message)
    
    if task.tool == "new_restaurant":

        memory = context.user_data.get(
            "restaurant_memory",
            {}
        )

        extracted = await extract_restaurant_data(
            memory,
            user_message
        )

        #memory update krta hai
        memory = merge_memory(
            memory,
            extracted
        )

        #updated memory save krta hai
        context.user_data[
            "restaurant_memory"
        ] = memory

        # check kya bach gya 
        missing_fields = get_missing_fields(
            memory
        )

        # bache hue fields k liye question puch
        question = await generate_followup_question(
            memory,
            missing_fields
        )

        await update.message.reply_text(
            question
        )

    elif task.tool == "suggest_places":
        memory = context.user_data.get(
            "conversation_memory",
            {}
        )

        query_number = 1

        response = await suggest_places(user_message, memory, query_number)

        context.user_data[
            "conversation_memory"
        ] = response["memory"]

        await update.message.reply_text(response["ai_response"])

    elif task.tool == "greeting":
        await update.message.reply_text(
            task.content
        )

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))

app.add_handler(
    MessageHandler(filters.TEXT, handle_message)
)

print("Bot started...")

app.run_polling()
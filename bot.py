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
        """
        🍕 Welcome to Always Hungry AI Agent!

        Humans forget. Stomachs don't.

        I remember your restaurant visits, food experiences, favorite dishes, and disappointments—so you never waste time wondering where to eat again.

        Just tell me where you've been, and when hunger strikes, I'll tell you where to go.

        Ready to feed my memory? 😋
        """
    )

async def suggest_places_operation(context, user_message, update):
    memory = context.user_data.get(
        "conversation_memory",
        {}
    )

    response = await suggest_places(user_message, memory)
    print("\nSuggest response", response)

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

async def new_restaurant_operation(user_message, context, update):
    memory = context.user_data.get(
        "restaurant_memory",
        {}
    )

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

async def handle_confirmation_response(context, user_message, update):
    active_task = context.user_data.get("active_task")

    if active_task == "suggest_places":
        await suggest_places_operation(context, user_message, update)
    elif active_task == "new_restaurant":
        await new_restaurant_operation(user_message, context, update)

    return


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_message = update.message.text.lower().strip()
    print("\n\nUser message: ", user_message)

    if user_message in [
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "no",
        "nah"
    ]:
        await handle_confirmation_response(context, user_message, update)
        return

    task = await task_selector(user_message)
    print("\nTop memory", context.user_data)

    # if "restaurant_memory" in context.user_data:
    # if task.tool == "new_restaurant" and "restaurant_memory" in context.user_data :

    #     new_restaurant_operation(user_message, context, update)
    #     return

    # if task.tool == "suggest_places" and "conversation_memory" in context.user_data:

    #     suggest_places_operation(context, user_message, update)
    #     return

    # task = await task_selector(user_message)

    context.user_data["active_task"] = task.tool
    
    if task.tool == "new_restaurant":

        await new_restaurant_operation(user_message, context, update)

    elif task.tool == "suggest_places":

        await suggest_places_operation(context, user_message, update)

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
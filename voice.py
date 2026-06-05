from openai import OpenAI

client = OpenAI()

async def get_transcript(voice, context):
    tg_file = await context.bot.get_file(voice.file_id)

    await tg_file.download_to_drive("voice.ogg")

    with open("voice.ogg", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )

    text = transcript.text
    print("Transcript", transcript)
    return text
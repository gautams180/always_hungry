import asyncio
import speech_recognition as sr
from openai import OpenAI
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
async_client = AsyncOpenAI()

def main():
    r = sr.Recognizer() # speech to text

    with sr.Microphone() as source: # Mic access
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2


        SYSTEM_PROMPT = f"""
            You are an expert voice agent. You are given transcript of what user has said using voice.
            Uou need to output as if you are an voice agent and whatever you speak will be converted back to audio using AI and played back to user.
        """

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        while(True):

            print("Speak something...")
            audio = r.listen(source)

            print("Processing audio... (STT)") # Speech to text
            stt = r.recognize_google(audio)

            print("You said: ",stt)

            messages.append({"role": "user", "content": stt})

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )

            print("AI Response ", response.choices[0].message.content)
            print(f"Token used: {response.usage.completion_tokens}")

            asyncio.run(tts(speech=response.choices[0].message.content))

main()
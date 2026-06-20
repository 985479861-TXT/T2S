import pyttsx3
import io
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from scipy.io import wavfile
from openai import OpenAI

# base set up
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')    

recognizer = sr.Recognizer()

# funcs
def setvoice(i=0):
    engine.setProperty('voice', voices[i].id)

def getvoices():
    for index, voice in enumerate(voices):
        print(f"Index: {index} | Name: {voice.name} | Gender: {voice.gender}")

def speak(text):
    engine.say(text)
    engine.runAndWait()

def stf(text, outpp):
    engine.save_to_file(text, outpp)
    engine.runAndWait()

def listenF(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription:")
            print(text)
            return text
        except sr.UnknownValueError:
            print("Audio was unclear.")
        except sr.RequestError:
            print("API service down.")

def listenMP(sample_rate=16_000):
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status Error: {status}")
        audio_bytes = (indata * 32767).astype(np.int16).tobytes()        
        try:
            audio_data = sr.AudioData(audio_bytes, sample_rate, 2)           
            # Request translation directly 
            text = recognizer.recognize_google(audio_data)
            print(f">> {text}")
            return text           
        except sr.UnknownValueError:
            pass # Ignore silent
        except sr.RequestError:
            print("[Connection issues reaching Google API]")
    # Open a direct continuous hardware recording stream
    with sd.InputStream(samplerate=sample_rate, channels=1, 
                        blocksize=int(sample_rate * 3), # Process in 3-seconds
                        callback=audio_callback):
        while True:
            sd.sleep(100) # Keep the thread awake

def ai(input, speak, key, train="You are a programming client handler."):
    client = OpenAI(api_key=key)

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": train},
        {"role": "user", "content": input}
    ],
    temperature=0.5  
    )
    chatgpt_reply = response.choices[0].message.content
    print(chatgpt_reply)
    speak(chatgpt_reply)

if __name__ == "__main__":
    pass
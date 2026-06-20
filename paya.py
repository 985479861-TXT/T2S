import pyttsx3
import io
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from scipy.io import wavfile
from openai import OpenAI
import pyaudiowpatch as pyaudio

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

def ai(input, speak, key, train="You are a programming client handler, handle the call with a client."):
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

def capture_pc():
    print("Listening...")
    with pyaudio.PyAudio() as p:
        try:
            default_speakers = p.get_default_wasapi_device()
        except OSError:
            return "Error: WASAPI audio devices not found. This requires Windows."
        
        loopback_idx = None
        for device in p.get_device_info_generator():
            if device["isLoopbackDevice"] and default_speakers["name"] in device["name"]:
                loopback_idx = device["index"]
                break
                
        if loopback_idx is None:
            return "Error: Default loopback device could not be isolated."

    recognizer = sr.Recognizer()
    
    # Adjust
    recognizer.pause_threshold = 3.0 
    
    # Configure energy levels
    recognizer.dynamic_energy_threshold = False 
    recognizer.energy_threshold = 1000 

    with sr.Microphone(device_index=loopback_idx) as source:
        try:
            # Adjusts for ambient noise baseline before capturing
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Listen until a silence gap exceeding pause_threshold occurs
            audio_data = recognizer.listen(source, timeout=None, phrase_time_limit=None)
            
            # Convert speech to text and return it
            return recognizer.recognize_google(audio_data)
            
        except sr.UnknownValueError:
            return "Error: System heard audio but it was unintelligible or just noise."
        except sr.RequestError as e:
            return f"Error: API connection issue ({e})"
        except KeyboardInterrupt:
            return "Error: Interrupted by user."


if __name__ == "__main__":
    setvoice(i=0)
    api = input("enter api key: ")
    print("ctrl + C to quit")
    while True:
        ai(input=capture_pc, speak=speak, key=api)

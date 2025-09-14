import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import google.generativeai as genai
from gtts import gTTS
import librosa
import os
from datetime import datetime

# Configuration
genai.configure(api_key="AIzaSyC0S9OWVvuXGCw6jMNnyfQV1hJD8uW0rQ8")
sample_rate = 44100
channels = 1
duration = 5
audio_folder = "recordings"
os.makedirs(audio_folder, exist_ok=True)

# Utility functions
def get_timestamped_filename():
    return f"userrequest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

def get_answer_timestamped_filename():
    return f"chatresponse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

def record_audio():
    st.info("Recording for 5 seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()
    wav_path = os.path.join(audio_folder, "temp.wav")
    write(wav_path, sample_rate, audio_data)

    mp3_path = os.path.join(audio_folder, get_timestamped_filename())
    sound = AudioSegment.from_wav(wav_path)
    sound.export(mp3_path, format="mp3")
    os.remove(wav_path)
    return mp3_path

def play_audio_file(file_path):
    sound = AudioSegment.from_mp3(file_path)
    play(sound)

def detect_audio_emotion(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    energy = np.mean(librosa.feature.rms(y=y))
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    if energy > 0.05 and pitch > 150:
        return "Excited"
    elif energy < 0.02:
        return "Sad or Tired"
    else:
        return "Neutral"

def transcribe_audio(mp3_path):
    wav_path = mp3_path.replace(".mp3", "_transcribe.wav")
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    os.remove(wav_path)
    return recognizer.recognize_google(audio, language="ar-AR")

def generate_genai_response(transcription):
    chatbot_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction='You are chatbot. you have to provide answer within maximum 3 lines')
    chat = chatbot_model.start_chat()
    response = chat.send_message(transcription)
    return response.text

def speak_text(text):
    tts = gTTS(text=text, lang='ar')
    mp3_path = os.path.join(audio_folder, get_answer_timestamped_filename())
    tts.save(mp3_path)
    return mp3_path

# Streamlit UI
st.title("🧠 Smart Voice ChatBot")

if st.button("🎙️ Ask Me"):
    mp3_path = record_audio()
    st.success(f"Audio saved: {os.path.basename(mp3_path)}")
    st.audio(mp3_path)

    if st.button("📝 Transcribe & Respond"):
        try:
            transcription = transcribe_audio(mp3_path)
            st.subheader("Transcription")
            st.write(transcription)

            response = generate_genai_response(transcription)
            st.subheader("GenAI Response")
            st.write(response)

            response_mp3 = speak_text(response)
            st.audio(response_mp3)

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.subheader("📂 Saved MP3 Files")
files = [f for f in os.listdir(audio_folder) if f.endswith(".mp3")]
selected_file = st.selectbox("Choose a file to play or analyze", files)

if selected_file:
    file_path = os.path.join(audio_folder, selected_file)
    st.audio(file_path)

    if st.button("▶️ Play Selected"):
        play_audio_file(file_path)

    if st.button("🤔 Detect Emotion"):
        emotion = detect_audio_emotion(file_path)
        st.info(f"Detected Emotion: {emotion}")

    if st.button("📝 Transcribe Selected"):
        try:
            transcription = transcribe_audio(file_path)
            st.subheader("Transcription")
            st.write(transcription)

            response = generate_genai_response(transcription)
            st.subheader("GenAI Response")
            st.write(response)

            response_mp3 = speak_text(response)
            st.audio(response_mp3)

        except Exception as e:
            st.error(f"Error: {str(e)}")

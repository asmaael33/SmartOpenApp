import streamlit as st
import random
import pyttsx3
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import speech_recognition as sr
from transformers import pipeline

# Load models
sentiment_model = pipeline("sentiment-analysis")
tts = pyttsx3.init()

# Define questions
questions = [
    "Comment te sens-tu aujourd'hui ?",
    "Qu'as-tu pensé de ta dernière réunion ?",
    "Comment décrirais-tu ta semaine ?",
    "Quel est ton ressenti par rapport à ton projet actuel ?",
    "As-tu apprécié ton dernier repas ?"
]

# Streamlit UI
st.title("🎙️ Détecteur d'Émotions Vocales")
st.write("Répondez à une question aléatoire avec votre voix. L'application détecte votre émotion à partir du ton et du texte.")

if st.button("Poser une question"):
    question = random.choice(questions)
    st.session_state["question"] = question
    tts.say(question)
    tts.runAndWait()
    st.write(f"🗨️ Question : {question}")

if "question" in st.session_state:
    duration = st.slider("Durée d'enregistrement (sec)", 3, 10, 5)
    if st.button("🎤 Enregistrer ma réponse"):
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
        sd.wait()
        sf.write("response.wav", recording, 44100)
        st.success("✅ Enregistrement terminé.")

        # Transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile("response.wav") as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language="fr-FR")
            except sr.UnknownValueError:
                text = ""
        st.write(f"📝 Réponse transcrite : {text}")

        # Text emotion
        if text:
            result = sentiment_model(text)[0]
            st.write(f"🧠 Émotion (texte) : {result['label']} (confiance : {result['score']:.2f})")

        # Audio emotion (simple logic)
        y, sr_audio = librosa.load("response.wav", sr=None)
        energy = np.mean(librosa.feature.rms(y=y))
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
        if energy > 0.05 and pitch > 150:
            audio_emotion = "Excité"
        elif energy < 0.02:
            audio_emotion = "Triste ou Fatigué"
        else:
            audio_emotion = "Neutre"
        st.write(f"🎼 Émotion (voix) : {audio_emotion}")

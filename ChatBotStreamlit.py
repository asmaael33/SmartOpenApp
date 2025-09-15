import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import speech_recognition as sr
import google.generativeai as genai
from gtts import gTTS
import librosa
import os
from datetime import datetime

# Setup
genai.configure(api_key="YOUR_API_KEY")  # Replace with your Gemini key
sample_rate = 44100
audio_folder = "recordings"
os.makedirs(audio_folder, exist_ok=True)

def get_timestamped_filename(prefix="userrequest"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

# Audio processor
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

    def save_recording(self):
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            wav_path = os.path.join(audio_folder, "temp.wav")
            write(wav_path, sample_rate, audio_data)
            mp3_path = os.path.join(audio_folder, get_timestamped_filename())
            sound = AudioSegment.from_wav(wav_path)
            sound.export(mp3_path, format="mp3")
            os.remove(wav_path)
            return mp3_path
        return None

def transcribe_audio(mp3_path):
    wav_path = mp3_path.replace(".mp3", "_transcribe.wav")
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    os.remove(wav_path)
    return recognizer.recognize_google(audio, language="ar-AR")

def generate_genai_response(text):
    chatbot_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction='You are a helpful assistant. Respond in 3 lines max.')
    chat = chatbot_model.start_chat()
    response = chat.send_message(text)
    return response.text

def speak_text(text):
    tts = gTTS(text=text, lang='ar')
    mp3_path = os.path.join(audio_folder, get_timestamped_filename("chatresponse"))
    tts.save(mp3_path)
    return mp3_path

def detect_audio_emotion(mp3_path):
    y, sr = librosa.load(mp3_path, sr=None)
    energy = np.mean(librosa.feature.rms(y=y))
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    if energy > 0.05 and pitch > 150:
        return "Excited"
    elif energy < 0.02:
        return "Sad or Tired"
    else:
        return "Neutral"

# Streamlit UI
st.title("🎙️ Smart Voice ChatBot")

ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    in_audio=True,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if ctx.audio_processor:
    if st.button("⏹️ Save & Process"):
        mp3_path = ctx.audio_processor.save_recording()
        if mp3_path:
            st.success(f"Saved: {os.path.basename(mp3_path)}")
            st.audio(mp3_path)

            try:
                transcription = transcribe_audio(mp3_path)
                st.subheader("📝 Transcription")
                st.write(transcription)

                emotion = detect_audio_emotion(mp3_path)
                st.subheader("🤔 Detected Emotion")
                st.info(emotion)

                response = generate_genai_response(transcription)
                st.subheader("🤖 GenAI Response")
                st.write(response)

                response_mp3 = speak_text(response)
                st.audio(response_mp3)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("No audio recorded yet.")

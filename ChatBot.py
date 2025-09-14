import tkinter as tk
from tkinter import messagebox, Listbox, Scrollbar
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play
import threading
import google.generativeai as genai
import os
from gtts import gTTS
from IPython.display import Audio, display
import speech_recognition as sp_r # Import speech_recognition
from datetime import datetime
import speech_recognition as sr
genai.configure(api_key="AIzaSyC0S9OWVvuXGCw6jMNnyfQV1hJD8uW0rQ8")
from mutagen.mp3 import MP3
import librosa

# Paramètres
sample_rate = 44100
channels = 1
wav_filename = "temp.wav"
recording = []
stream = None
mp3_filename = ""
audio_folder = "." 
duration = 5 

# 🎼 Audio Emotion Detection (simple MFCC + threshold logic)
def detect_audio_emotion(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    energy = np.mean(librosa.feature.rms(y=y))
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))

    # Simple thresholds (can be replaced with a trained model)
    if energy > 0.05 and pitch > 150:
        messagebox.showinfo("Excited")
    elif energy < 0.02:
        messagebox.showinfo("Sad or Tired")
    else:
        messagebox.showinfo("Neutral")

def get_timestamped_filename():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    return f"userrequest_{timestamp}.mp3"

def get_answer_timestamped_filename():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    return f"chatresponse_{timestamp}.mp3"

def callback(indata, frames, time, status):
    if status:
        print(status)
    recording.append(indata.copy())

def start_recording():
    global stream, recording, mp3_filename
    recording = []
    record_button.config(text="🔴 Recording…", bg="red", state="disabled")
    stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16', callback=callback)
    stream.start()

def stop_recording():
    global stream, mp3_filename
    record_button.config(text="🎙️ AskMe", bg="lightgreen")
    try:
        if stream:
            stream.stop()
            stream.close()
            stream = None

            audio_data = np.concatenate(recording, axis=0)
            wav_filename = "temp.wav"
            write(wav_filename, sample_rate, audio_data)

            mp3_filename = get_timestamped_filename()
            sound = AudioSegment.from_wav(wav_filename)
            sound.export(mp3_filename, format="mp3")
            os.remove(wav_filename)

            messagebox.showinfo("Saved", f"Audio saved as {mp3_filename}")
            refresh_mp3_list()
        else:
            messagebox.showwarning("Warning", "Recording was not started.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def play_audio(file=None):
    try:
        target = file or mp3_filename
        if target and os.path.exists(target):
            sound = AudioSegment.from_mp3(target)
            threading.Thread(target=play, args=(sound,), daemon=True).start()
        else:
            messagebox.showwarning("Warning", "No audio file found to play.")
    except Exception as e:
        messagebox.showerror("Error", f"Playback failed: {str(e)}")

def refresh_mp3_list():
    listbox.delete(0, tk.END)
    files = [f for f in os.listdir(audio_folder) if f.endswith(".mp3")]
    for f in sorted(files):
        listbox.insert(tk.END, f)

def on_select(event):
    selection = event.widget.curselection()
    if selection:
        index = selection[0]
        filename = listbox.get(index)
        play_audio(filename)
        detect_audio_emotion(filename)



def transcribe_audio(file=None):
    try:
        target = file or mp3_filename
        if not target or not os.path.exists(target):
            messagebox.showwarning("Warning", "No MP3 file found to transcribe.")
            return

        # Convert MP3 to WAV
        sound = AudioSegment.from_mp3(target)
        wav_path = "temp_for_transcription.wav"
        sound.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        transcription = recognizer.recognize_google(audio, language="ar-AR")
        os.remove(wav_path)

        # Show transcription
        messagebox.showinfo("Transcription", transcription)

        # GenAI response
        response = generate_genai_response(transcription)
        messagebox.showinfo("GenAI Response", response)

        speak_text(response)

    except sr.UnknownValueError:
        messagebox.showerror("Transcription Error", "Could not understand the audio.")
    except sr.RequestError as e:
        messagebox.showerror("API Error", f"Speech recognition failed: {e}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def generate_genai_response(transcription):
    try:
        os.environ["API_KEY"] = 'AIzaSyC0S9OWVvuXGCw6jMNnyfQV1hJD8uW0rQ8'
        genai.configure(api_key=os.environ["API_KEY"])
        instruction='You are chatbot. you have to provide answer within maximum 3 lines'
        # Rename the GenerativeModel instance to avoid conflict
        chatbot_model = genai.GenerativeModel('gemini-1.5-flash-latest',system_instruction=instruction)
        chat=chatbot_model.start_chat()
        response = chat.send_message(transcription) # Use the chat object
        prompt = f"L'utilisateur a dit : \"{transcription}\". Réponds intelligemment à cette requête."
        
        return response.text
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

def transcribe_selected():
    selection = listbox.curselection()
    if selection:
        filename = listbox.get(selection[0])
        transcribe_audio(filename)
    else:
        messagebox.showwarning("Warning", "Please select an MP3 file to transcribe.")


def speak_text(text):
    langChatBot = 'ar' # Language for gTTS (Spanish)
    tts = gTTS(text=text, lang=langChatBot)
    mp3_filename = get_answer_timestamped_filename()
    tts.save(mp3_filename)
    play_audio(mp3_filename)


def refresh_mp3_list():
    listbox.delete(0, tk.END)
    files = [f for f in os.listdir(audio_folder) if f.endswith(".mp3")]
    for f in sorted(files):
        listbox.insert(tk.END, f)


# GUI setup
root = tk.Tk()
root.title("ChatterBot")
root.geometry("360x400")

label = tk.Label(root, text="ChatBot", font=("Arial", 12))
label.pack(pady=10)

record_button = tk.Button(root, text="🎙️ AskMe", command=start_recording, font=("Arial", 12), bg="lightgreen")
record_button.pack(pady=5)

stop_button = tk.Button(root, text="⏹️ StopRecording", command=stop_recording, font=("Arial", 12), bg="lightcoral")
stop_button.pack(pady=5)

#play_button = tk.Button(root, text="▶️ PlayLast", command=lambda: play_audio(), font=("Arial", 12), bg="lightblue")
#play_button.pack(pady=5)


detect_emotion_button = tk.Button(root, text="🤔 DetectEmotion", command=lambda: detect_audio_emotion(), font=("Arial", 12), bg="lightblue")
detect_emotion_button.pack(pady=5)


refresh_button = tk.Button(root, text="🔄 RefreshList", command=refresh_mp3_list, font=("Arial", 12), bg="lightgray")
refresh_button.pack(pady=5)


transcribe_selected_button = tk.Button(root, text="📝 AnswerRequest", font=("Arial", 12), bg="lightyellow")
transcribe_selected_button.pack(pady=5)
transcribe_selected_button.config(command=transcribe_selected)



list_label = tk.Label(root, text="📂 Saved MP3 Files", font=("Arial", 11))
list_label.pack(pady=10)

scrollbar = Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

listbox = Listbox(root, width=40, height=10, font=("Arial", 10), yscrollcommand=scrollbar.set)
listbox.pack(pady=5)
listbox.bind('<<ListboxSelect>>', on_select)
scrollbar.config(command=listbox.yview)

refresh_mp3_list()

root.mainloop()
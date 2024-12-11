import streamlit as st
import librosa
import soundfile as sf
import os
import numpy as np
from scipy.signal import medfilt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langchain_community.llms import Ollama

# Download NLTK data
nltk.download('punkt')

# Function to preprocess audio
def preprocess_audio(file_name, output_dir="preprocessed"):
    audio, sr = librosa.load(file_name, mono=True)
    audio = audio / np.max(np.abs(audio))  # Normalize the audio
    audio_denoised = medfilt(audio, kernel_size=3)  # Basic noise reduction
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    preprocessed_file = os.path.join(output_dir, f"preprocessed_{os.path.basename(file_name)}")
    sf.write(preprocessed_file, audio_denoised, sr)
    return preprocessed_file

# Function to split audio into segments
def split_audio(file_name, output_dir="out", segment_duration=30):
    audio, sr = librosa.load(file_name)
    audio_slow = librosa.effects.time_stretch(audio, rate=0.95)  # Slow down the audio
    buffer = segment_duration * sr
    samples_total = len(audio_slow)
    samples_wrote = 0
    counter = 1
    file_base_name = os.path.splitext(os.path.basename(file_name))[0]
    split_dir = os.path.join(output_dir, file_base_name)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    while samples_wrote < samples_total:
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote
        block = audio_slow[samples_wrote: (samples_wrote + buffer)]
        out_filename = os.path.join(split_dir, f"split_{counter}.wav")
        sf.write(out_filename, block, sr)
        counter += 1
        samples_wrote += buffer
    return split_dir

# Load speech-to-text model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=20,
    return_timestamps=True,
    torch_dtype=torch.float16,
    device=device,
)

# Function to perform productivity analysis
def evaluate_productivity(transcription):
    productivity_keywords = ["decision", "transcribe", "program", "task", "action", "plan", "solution", "agenda", "discuss"]
    productive_segments = []
    sentences = sent_tokenize(transcription)
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in productivity_keywords):
            productive_segments.append(sentence)
    return productive_segments

# Sentiment analysis function
def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    sentiment = "positive" if sentiment_scores["compound"] >= 0.05 else "negative" if sentiment_scores["compound"] <= -0.05 else "neutral"
    return sentiment

# Streamlit UI
st.title("Audio to Text Meeting Analyzer")

# Upload audio file
audio_file = st.file_uploader("Upload your meeting audio file", type=["mp3", "wav", "flac"])

if audio_file:
    st.audio(audio_file, format='audio/wav')

    # Save the uploaded file
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Step 1: Preprocess the audio
    preprocessed_file = preprocess_audio("uploaded_audio.wav")
    st.write(f"Audio preprocessed and saved at: {preprocessed_file}")

    # Step 2: Split the preprocessed audio
    split_dir = split_audio(preprocessed_file)
    st.write(f"Audio split into segments at: {split_dir}")

    # Step 3: Perform speech-to-text conversion
    transcription_ls = []
    audio_ls = sorted(os.listdir(split_dir))
    for audio in audio_ls:
        result = pipe(os.path.join(split_dir, audio))
        transcription_ls.append(result["text"])

    transcription = "\n\n".join(transcription_ls)
    st.subheader("Transcription:")
    st.text_area("Transcription", transcription, height=300)

    # Step 4: Generate meeting minutes
    llm = Ollama(model="mistral")
    final_note = llm.invoke(f"Give point-wise minutes of the meeting from this text: {transcription}")
    st.subheader("Meeting Minutes:")
    st.text_area("Minutes", final_note, height=300)

    # Step 5: Evaluate productivity
    productive_segments = evaluate_productivity(transcription)
    st.subheader("Productive Segments:")
    st.write("\n".join(productive_segments) if productive_segments else "No productive segments found.")

    # Step 6: Sentiment analysis
    sentiment = sentiment_analysis(transcription)
    st.subheader("Sentiment Analysis:")
    st.write(f"Overall Sentiment: {sentiment}")

    sentiment_per_segment = [f"Sentiment for Segment: {sentiment_analysis(segment)}" for segment in transcription_ls]
    st.write("\n".join(sentiment_per_segment))

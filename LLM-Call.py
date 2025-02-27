# pip install openai-whisper yt-dlp ffmpeg-python transformers torch numpy

import subprocess
import numpy as np
import ffmpeg
import whisper
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from io import BytesIO
import torch
from huggingface_hub import login

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        
    torch_dtype=torch.float16 
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 1: Stream YouTube audio using yt_dlp
def stream_youtube_audio(video_url):
    """
    Streams audio from a YouTube video without downloading it.
    """
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--no-playlist",
        "-o", "-",  
        video_url
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout.read()

# Step 2: Convert audio stream to NumPy array
def audio_stream_to_numpy(audio_bytes):
    """
    Converts an audio byte stream to a NumPy array using ffmpeg.
    """
    try:
        out, _ = (
            ffmpeg.input("pipe:0")
            .output("pipe:1", format="wav", acodec="pcm_s16le", ac=1, ar="16000")
            .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
        )
        audio_data = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        return audio_data
    except ffmpeg.Error as e:
        print("FFmpeg error:", e)
        return None

# Step 3: Transcribe audio using Whisper
def transcribe_audio_numpy(audio_data):
    """
    Transcribes audio data using Whisper.
    """
    model_whisper = whisper.load_model("base")
    result = model_whisper.transcribe(audio_data)
    print("Transcription completed.")
    return result["text"]

# Step 4: Summarization
def summarize_text(transcription, max_tokens=512):
    """
    Summarizes the transcription to reduce context size before question generation.
    """
    if len(transcription.split()) < 100:
        return transcription

    summary = summarizer(transcription, max_length=max_tokens, min_length=100, do_sample=False)
    return summary[0]['summary_text']

# Step 5: Generate questionnaire
def generate_questionnaire(summary):
    prompt = f"""
You are a professional questionnaire generator reputed for generating diverse questionnaires, given any
information sample.

The questionnaire you generate must contain:
1. Three simple multiple-choice questions (each with 4 options).
2. One moderately difficult multiple-choice question (4 options).
3. Two simple open-ended questions.
4. Three moderately difficult open-ended questions.
5. One hard scenario-based open-ended question.

Make sure to cover each and every type of question mentioned.
Nothing else, no code. Stick strictly to the provided context.
Also, provide the questions in a structured, well-formatted, sequential manner.
Start question sections with ### Multiple-Choice Questions etc.
Generate a well-structured questionnaire based on the following content:
"{summary}"
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=2000,  
        temperature=0.2,
        top_p=0.8,
        repetition_penalty=1.1,
        do_sample=True
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.strip()   

# Step 6: Full pipeline
def process_stream(video_url):
    print("Streaming audio...")
    audio_bytes = stream_youtube_audio(video_url)
    if not audio_bytes:
        print("Error: Unable to fetch audio.")
        return

    print("Converting audio stream to NumPy array...")
    audio_data = audio_stream_to_numpy(audio_bytes)
    if audio_data is None:
        print("Error: Unable to process audio data.")
        return

    print("Transcribing audio...")
    transcription = transcribe_audio_numpy(audio_data)
    if not transcription:
        print("Error: Transcription failed.")
        return

    print("Summarizing transcription...")
    summary = summarize_text(transcription)

    print("Generating questionnaire...")
    questionnaire = generate_questionnaire(summary)
    
    print("\nGenerated Questionnaire:")
    print(questionnaire)
    return questionnaire

youtube_url = "https://www.youtube.com/watch?v=L2YiNu22saU"  # Replace with your YouTube video link
process_stream(youtube_url)

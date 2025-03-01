#pip install openai-whisper yt-dlp ffmpeg-python transformers torch numpy reportlab

import subprocess
import numpy as np
import ffmpeg
import whisper
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from io import BytesIO
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from IPython.display import FileLink
import textwrap

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
    model_whisper = whisper.load_model("tiny")
    result = model_whisper.transcribe(audio_data)
    print("Transcription completed.")
    return result["text"]

def chunk_text(text, max_words=512):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks


def final_summarize_cpu(text, max_length=150, min_length=50):
    # Create a CPU summarizer instance
    cpu_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    # Explicitly tokenize with truncation; set a max_length that fits the model (e.g. 1024 tokens)
    inputs = cpu_summarizer.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    # Generate summary
    summary_ids = cpu_summarizer.model.generate(**inputs, max_length=max_length, min_length=min_length, do_sample=False)
    final_summary = cpu_summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return final_summary
    
# Step 4: Summarization with chunking for long transcriptions
def summarize_text(transcription, max_tokens=512):
    if len(transcription.split()) < 1000:
        summary = summarizer(transcription, max_length=max_tokens, min_length=100, do_sample=False)
        return summary[0]['summary_text']

    # For longer transcriptions, split into chunks, summarize each, then combine summaries.
    print("Transcription too long; splitting into chunks...")
    chunks = chunk_text(transcription, max_words=512)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        try:
            summary = summarizer(chunk, max_length=max_tokens, min_length=100, do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {e}")
            chunk_summaries.append("")  # Optionally handle failed chunks
    combined_summary = " ".join(chunk_summaries)
    # Use the CPU-based final summarization with explicit tokenization & truncation.
    try:
        final_summary = final_summarize_cpu(combined_summary, max_length=150, min_length=50)
        return final_summary
    except Exception as e:
        print(f"Error in final summarization: {e}")
        return combined_summary

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
    #print(output_text)
    print("Questionnaire generation completed.")
    #return clean_questionnaire(output_text)
    return clean_questionnaire(output_text)
   
#Step 6: Clean output
def clean_questionnaire(raw_text):
    match = re.search(r"(### Multiple-Choice Questions.*?)$", raw_text, re.DOTALL)
    cleaned_text = match.group(1) if match else raw_text  
    #cleaned_text = re.sub(r"(### Multiple-Choice Questions.*?)\s*Generate a well-structured questionnaire.*$", r"\1", cleaned_text, flags=re.DOTALL)
    return cleaned_text.strip()

# Function to save text as a PDF using ReportLab
def save_text_as_pdf(text, filename):
    pdf = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    margin = 50
    available_width = width - 2 * margin
    text_object = pdf.beginText(margin, height - margin)
    text_object.setFont("Helvetica", 12)

    max_chars_per_line = 100

    for paragraph in text.split("\n"):
        # Wrap the paragraph using textwrap
        wrapped_lines = textwrap.wrap(paragraph, width=max_chars_per_line)
        if not wrapped_lines:
            # For empty lines, add a blank line
            text_object.textLine("")
        for line in wrapped_lines:
            text_object.textLine(line)
            # Check for page break
            if text_object.getY() < margin:
                pdf.drawText(text_object)
                pdf.showPage()
                text_object = pdf.beginText(margin, height - margin)
                text_object.setFont("Helvetica", 12)    
    
    pdf.drawText(text_object)
    pdf.save()

# Step 7: Full pipeline
def process_stream(video_url, output_pdf="questionnaire.pdf"):
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

    print("Converting questionnaire to PDF...")
    save_text_as_pdf(questionnaire, output_pdf)

    print(f"\nPDF generated: {output_pdf}")
    return output_pdf
    
youtube_url = "https://www.youtube.com/watch?v=CVfnkM44Urs&list=PLU630Cd0ZQCMeQiSvU7DJmDJDitdE7m7r&index=2"  # Replace with your YouTube video link
pdf_file = process_stream(youtube_url)

if pdf_file:
    # Display a download link in the notebook (Kaggle environment)
    display(FileLink(pdf_file))

import os
import streamlit as st
from pytube import YouTube
from whisper import load_model
from gemini_pro import GeminiProLLM
from dotenv import load_dotenv
from collections import Counter
import math
import re

# Load environment variables
load_dotenv()
GEMINI_PRO_API_KEY = os.getenv('GEMINI_PRO_API_KEY')

def download_audio(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    audio_path = stream.download(filename='audio.mp4')
    return audio_path

def transcribe_audio(audio_path, model_name='base'):
    model = load_model(model_name)
    transcription = model.transcribe(audio_path)
    return transcription['text']

def summarize_text(transcript, paragraphs_count, char_count, api_key):
    gemini = GeminiProLLM(api_key=api_key)
    summary = gemini.summarize(text=transcript, paragraphs_count=paragraphs_count, char_count=char_count)
    return summary

def extract_key_phrases(text):
    words = re.findall(r'\w+', text.lower())
    word_counts = Counter(words)
    most_common = word_counts.most_common(10)
    return [word for word, _ in most_common]

def calculate_metrics(text):
    words = text.split()
    word_count = len(words)
    reading_time = math.ceil(word_count / 200)  # Average reading speed is 200 WPM
    key_phrases = extract_key_phrases(text)
    return word_count, reading_time, key_phrases

st.title('YouTube Video Summarizer')

youtube_url = st.text_input('Enter YouTube URL:')
paragraphs_count = st.number_input('Enter number of paragraphs:', min_value=1, value=3)
char_count = st.number_input('Enter maximum character count:', min_value=100, value=500)

if st.button('Generate Summary'):
    if youtube_url:
        with st.spinner('Downloading audio...'):
            audio_path = download_audio(youtube_url)
        
        with st.spinner('Transcribing audio...'):
            transcript = transcribe_audio(audio_path)
        
        with st.spinner('Generating summary...'):
            summary = summarize_text(transcript, paragraphs_count, char_count, GEMINI_PRO_API_KEY)
            st.success('Summary generated successfully!')
            st.write(summary)
            
            # Display transcript in a collapsible section
            with st.expander("View Transcript"):
                st.write(transcript)
            
            # Provide download option for the summary
            st.download_button('Download Summary', summary, file_name='summary.txt')

            # Calculate and display additional metrics
            word_count, reading_time, key_phrases = calculate_metrics(transcript)
            st.subheader('Additional Metrics')
            st.write(f'Word Count: {word_count}')
            st.write(f'Reading Time: {reading_time} minute(s)')
            st.write(f'Key Phrases: {", ".join(key_phrases)}')
    else:
        st.error('Please enter a valid YouTube URL.')

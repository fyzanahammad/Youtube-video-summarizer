import streamlit as st
import yt_dlp
from whisper import load_model
from gemini_pro import GeminiProLLM

def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return 'audio.mp3'

def transcribe_audio(audio_path, model):
    transcription = model.transcribe(audio_path)
    return transcription['text']

def summarize_text(transcript, paragraphs_count, char_count):
    gemini = GeminiProLLM()
    summary = gemini.summarize(text=transcript, paragraphs_count=paragraphs_count, char_count=char_count)
    return summary

st.title('YouTube Video Summarizer')

youtube_url = st.text_input('Enter YouTube URL:')
paragraphs_count = st.number_input('Enter number of paragraphs:', min_value=1, value=3)
char_count = st.number_input('Enter maximum character count:', min_value=100, value=500)

if st.button('Generate Summary'):
    if youtube_url:
        with st.spinner('Downloading audio...'):
            audio_path = download_audio(youtube_url)
        
        with st.spinner('Transcribing audio...'):
            model = load_model('base')
            transcript = transcribe_audio(audio_path, model)
        
        with st.spinner('Generating summary...'):
            summary = summarize_text(transcript, paragraphs_count, char_count)
            st.success('Summary generated successfully!')
            st.write(summary)
    else:
        st.error('Please enter a valid YouTube URL.')

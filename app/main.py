# --- Combined Main Program for WebCast ---
import streamlit as st
import os
from app.search.search_trigger import needs_live_data_openai
from webcast_engine_openai import get_llm_summary
from merged_search import merged_search
from app.speach.text_to_speech import text_to_mp3_file, list_saved_files


def handle_send(prompt):
    """
    Handles user prompt: decides if live data is needed, fetches web data if required, summarizes with model.
    """
    needs_live = needs_live_data_openai(prompt)
    if needs_live:
        st.info("Fetching live data from the web...")
        paragraphs = merged_search(prompt)
        context = "\n".join(paragraphs)
        answer = get_llm_summary(prompt, context)
    else:
        st.info("Answering from model's existing knowledge...")
        answer = get_llm_summary(prompt, None)
    return answer

def on_voice(text):
    """Converts text to speech and returns MP3 file path."""
    mp3_folder = "mp3"
    mp3_path = text_to_mp3_file(text, mp3_folder)
    if mp3_path:
        st.success(f"MP3 file generated: {mp3_path}")
    else:
        st.error("Failed to generate MP3 file.")
    return mp3_path

def main():
    st.set_page_config(page_title="WebCast", layout="wide")
    st.title("WebCast: AI-powered Search & Summarization")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mp3_url" not in st.session_state:
        st.session_state.mp3_url = None

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Type your question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Processing..."):
            answer = handle_send(prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        # TTS button
        if st.button("Convert answer to speech"):
            mp3_path = on_voice(answer)
            if mp3_path:
                st.session_state.mp3_url = mp3_path

    # Sidebar: MP3 files
    with st.sidebar:
        st.header("Audio Files")
        mp3_folder = "mp3"
        files = list_saved_files(mp3_folder)
        if files:
            selected = st.selectbox("Select MP3 to play", files)
            if selected:
                st.audio(selected, format="audio/mp3")
        else:
            st.write("No audio files found.")

if __name__ == "__main__":
    main()
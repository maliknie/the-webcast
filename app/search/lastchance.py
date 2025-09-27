import streamlit as st
import time
import os
import uuid
import sys

# Add the parent directory to the path to import from app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main pipeline and text-to-speech functions
from main import webcast_pipeline
from speach.text_to_speech import stream_text_to_speech, text_to_mp3_file


## --- REAL PIPELINE FUNCTIONS ---##

# Real function that processes user prompts through the webcast pipeline
def handle_send(prompt):
    try:
        # Run the webcast pipeline without voice generation first
        result = webcast_pipeline(prompt, enable_voice=False)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Store additional pipeline information in session state
        st.session_state.pipeline_info = {
            "search_performed": result.get("search_performed", False),
            "context_used": result.get("context_used", False),
            "text_response": result["text_response"]
        }
        
        # Return the text response
        return result["text_response"]
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Real function that converts text to speech using stream_text_to_speech
def on_voice(text=""):
    print(f"[DEBUG] on_voice called with text: {text[:100] if text else 'None'}...")
    if not text:
        st.error("No text provided for voice conversion.")
        return None
    
    try:
        # Create mp3 folder if it doesn't exist - use absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mp3_folder = os.path.join(current_dir, "..", "mp3")
        mp3_folder = os.path.abspath(mp3_folder)
        
        if not os.path.exists(mp3_folder):
            os.makedirs(mp3_folder)
            print(f"[DEBUG] Created mp3 folder: {mp3_folder}")
        
        print(f"[DEBUG] Using mp3 folder: {mp3_folder}")
        
        # Generate audio file using text_to_mp3_file
        mp3_path = text_to_mp3_file(
            text=text,
            out_dir=mp3_folder,
            filename_prefix="webcast_response"
        )
        
        print(f"[DEBUG] Generated mp3_path: {mp3_path}")
        
        if mp3_path and os.path.exists(mp3_path):
            file_size = os.path.getsize(mp3_path)
            print(f"[DEBUG] File exists, size: {file_size} bytes")
            return mp3_path
        else:
            print(f"[DEBUG] File not found at: {mp3_path}")
            return None
            
    except Exception as e:
        error_msg = f"Error generating audio: {str(e)}"
        print(f"[DEBUG] Exception: {error_msg}")
        st.error(error_msg)
        return None

## --------------- END REAL FUNCTIONS ---------------##

class ChatUI:
    def __init__(self, on_send=handle_send, on_voice=on_voice):
        self.on_send = on_send
        self.on_voice = on_voice
        self._init_session_state()

    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "assistant_response" not in st.session_state:
            st.session_state.assistant_response = None
        if "current_audio" not in st.session_state:
            st.session_state.current_audio = None
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False
        if "pipeline_info" not in st.session_state:
            st.session_state.pipeline_info = None
        if "audio_generation_trigger" not in st.session_state:
            st.session_state.audio_generation_trigger = False
        if "last_message_index" not in st.session_state:
            st.session_state.last_message_index = -1

    def render(self):
        st.title("WebCast")

        # Create columns for reset button and audio player
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Reset Messages"):
                st.session_state.messages = []
                st.session_state.assistant_response = None
                st.session_state.current_audio = None
                st.session_state.is_processing = False
                st.session_state.pipeline_info = None
                st.session_state.audio_generation_trigger = False
                st.session_state.last_message_index = -1

        # Persistent audio player at the top
        with col2:
            if st.session_state.current_audio and os.path.exists(st.session_state.current_audio):
                st.audio(st.session_state.current_audio, format="audio/mp3")
                st.caption(f"🎵 {os.path.basename(st.session_state.current_audio)}")

        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(f'<p style="font-family: Arial, sans-serif; color: #e0e0e0; font-size: 16px;">{message["content"]}</p>', unsafe_allow_html=True)
                
                # Add Convert to Speech button for assistant messages
                if message["role"] == "assistant" and message["content"].startswith("Apertus:"):
                    # Extract the actual response text
                    response_text = message["content"][9:].strip()  # Remove "Apertus: " prefix
                    if st.button("🎵 Convert to Speech", key=f"voice_{i}"):
                        with st.spinner("🎵 Generating audio..."):
                            mp3_path = self.on_voice(response_text)
                            if mp3_path:
                                st.session_state.current_audio = mp3_path
                                st.success("✅ Audio generated successfully!")
                                st.rerun()

        # Accept user input
        if prompt := st.chat_input("Type something..."):
            with st.chat_message("user"):
                st.markdown(f"User: {prompt}")
            st.session_state.messages.append({"role": "user", "content": f"User: {prompt}"})

            with st.spinner('Apertus is thinking...'):
                assistant_response = self.on_send(prompt)
            st.session_state.assistant_response = assistant_response
            
            # Store the index of this message for reference
            st.session_state.last_message_index = len(st.session_state.messages)

            with st.chat_message("assistant"):
                st.markdown(f'<p style="font-family: Arial, sans-serif; color: #e0e0e0; font-size: 16px;">Apertus: {assistant_response}</p>', unsafe_allow_html=True)
                
                # Display pipeline information if available
                if st.session_state.pipeline_info:
                    with st.expander("🔍 Pipeline Information", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.session_state.pipeline_info["search_performed"]:
                                st.success("✓ Web search performed")
                            else:
                                st.info("ℹ️ No web search needed")
                        with col2:
                            if st.session_state.pipeline_info["context_used"]:
                                st.success("✓ Live data used")
                            else:
                                st.info("ℹ️ Using existing knowledge")
                
                # Convert to speech button for the new response
                if st.button("🎵 Convert to Speech", key=f"voice_new_{st.session_state.last_message_index}"):
                    with st.spinner("🎵 Generating audio..."):
                        mp3_path = self.on_voice(assistant_response)
                        if mp3_path:
                            st.session_state.current_audio = mp3_path
                            st.success("✅ Audio generated successfully!")
                            st.rerun()

            st.session_state.messages.append({"role": "assistant", "content": f"Apertus: {assistant_response}"})

        # Sidebar to display MP3 files from the folder
        with st.sidebar:
            st.header("🎵 Audio Library")
            
            # Define the MP3 folder relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mp3_folder = os.path.join(current_dir, "..", "mp3")
            mp3_folder = os.path.abspath(mp3_folder)
            
            if not os.path.exists(mp3_folder):
                os.makedirs(mp3_folder)
            
            mp3_files = [f for f in os.listdir(mp3_folder) if f.endswith(".mp3")]
            
            if mp3_files:
                # Sort files by modification time (newest first)
                mp3_files.sort(key=lambda x: os.path.getmtime(os.path.join(mp3_folder, x)), reverse=True)
                
                st.subheader("📁 Recent Audio Files")
                for i, file in enumerate(mp3_files[:10]):  # Show last 10 files
                    file_path = os.path.join(mp3_folder, file)
                    # Display file name with timestamp
                    file_time = os.path.getmtime(file_path)
                    from datetime import datetime
                    timestamp = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M")
                    
                    if st.button(f"🎵 {file[:30]}... ({timestamp})", key=f"play_sidebar_{i}"):
                        st.session_state.current_audio = file_path
                        st.rerun()
                
                # Add clear all button
                if st.button("🗑️ Clear All Audio Files"):
                    for file in mp3_files:
                        try:
                            os.remove(os.path.join(mp3_folder, file))
                        except:
                            pass
                    st.success("All audio files cleared!")
                    st.session_state.current_audio = None
                    st.rerun()
            else:
                st.info("No audio files yet. Generate speech from a message first.")

if __name__ == "__main__":
    chat_ui = ChatUI()  
    chat_ui.render()

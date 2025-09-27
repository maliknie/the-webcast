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
            st.success(f"🎵 Audio generated successfully! ({file_size} bytes)")
            return mp3_path
        else:
            st.error("Failed to generate audio file.")
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
        if "mp3_url" not in st.session_state:
            st.session_state.mp3_url = None
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False
        if "pipeline_info" not in st.session_state:
            st.session_state.pipeline_info = None

    def render(self):
        st.title("WebCast")

        if st.button("Reset Messages"):
            st.session_state.messages = []
            st.session_state.assistant_response = None
            st.session_state.mp3_url = None
            st.session_state.is_processing = False
            st.session_state.pipeline_info = None

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f'<p style="font-family: Arial, sans-serif; color: #e0e0e0; font-size: 16px;">{message["content"]}</p>', unsafe_allow_html=True)

        # Accept user input
        if prompt := st.chat_input("Type something..."):
            with st.chat_message("user"):
                st.markdown(f"User: {prompt}")
            st.session_state.messages.append({"role": "user", "content": f"User: {prompt}"})

            with st.spinner('Apertus is thinking...'):
                assistant_response = self.on_send(prompt)
            st.session_state.assistant_response = assistant_response

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
                
                # Display audio player if audio is available
                if st.session_state.mp3_url and os.path.exists(st.session_state.mp3_url):
                    st.audio(st.session_state.mp3_url, format="audio/mp3")
                    st.success(f"🎵 Playing: {os.path.basename(st.session_state.mp3_url)}")

            st.session_state.messages.append({"role": "assistant", "content": f"Apertus: {assistant_response}"})

            # Convert to speech button - always show it
            button_clicked = st.button("🎵 Convert to Speech")
            if button_clicked:
                print(f"[DEBUG] Button clicked, assistant_response: {assistant_response[:100] if assistant_response else 'None'}...")
                st.write("🔊 **Button clicked, assistant...**")  # Show in UI instead of just terminal
                st.session_state.is_processing = True
                
                with st.spinner("🎵 Generating audio..."):
                    mp3_path = self.on_voice(assistant_response)
                    if mp3_path:
                        st.session_state.mp3_url = mp3_path
                        st.success("🎵 Audio generated! Scroll up to play it.")
                        st.rerun()  # Refresh to show the audio player
                
                st.session_state.is_processing = False

        if st.session_state.is_processing:
            st.spinner("Processing...")

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
                for i, file in enumerate(mp3_files[:5]):  # Show last 5 files
                    file_path = os.path.join(mp3_folder, file)
                    if st.button(f"🎵 {file}", key=f"play_{i}"):
                        st.session_state.mp3_url = file_path
                        st.rerun()
            else:
                st.info("No audio files yet. Generate speech from a message first.")

if __name__ == "__main__":
    chat_ui = ChatUI()  
    chat_ui.render()

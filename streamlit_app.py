import streamlit as st
import time
import os
import uuid  


## --- IGNORE: These are default functions for testing ---##

# Default function that simulates assistant response, for testing
def handle_send(prompt):
    time.sleep(2)  
    return f"This is a delayed response to: {prompt}"

# Default function that simulates converting text to speech and generating an MP3 path, for testing
def on_voice(text= ""):
    time.sleep(2) 
    mp3_folder = "mp3"  # Folder where MP3 files are stored

    if not os.path.exists(mp3_folder):
        os.makedirs(mp3_folder)

    # Create a random file name for each MP3 file
    mp3_filename = f"response_{uuid.uuid4().hex}.mp3"
    mp3_path = os.path.join(mp3_folder, mp3_filename)

    # Create a mock MP3 file
    with open(mp3_path, "w") as f:
        f.write("This is a mock MP3 file. Replace with actual audio generation logic.")

    if os.path.exists(mp3_path):
        st.write(f"MP3 file generated successfully. Check your sidebar to play it.")
        return mp3_path
    else:
        st.write(f"Failed to save MP3 file.")
        return None

## --------------- END IGNORE ---------------##

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
        if "is_button_clicked" not in st.session_state:
            st.session_state.is_button_clicked = False

    def render(self):
        st.title("WebCast")

        if st.button("Reset Messages"):
            st.session_state.messages = []
            st.session_state.assistant_response = None
            st.session_state.mp3_url = None
            st.session_state.is_processing = False
            st.session_state.is_button_clicked = False

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f'<p style="font-family: Arial, sans-serif; color: #e0e0e0; font-size: 16px;">{message["content"]}</p>', unsafe_allow_html=True)

        # Accept user input
        if prompt := st.chat_input("Type something..."):
            with st.chat_message("user"):
                st.markdown(f"User: {prompt}")
            st.session_state.messages.append({"role": "user", "content": f"User: {prompt}"})

            with st.spinner('Assistant is typing...'):
                assistant_response = self.on_send(prompt)
            st.session_state.assistant_response = assistant_response

            with st.chat_message("assistant"):
                st.markdown(f'<p style="font-family: Arial, sans-serif; color: #e0e0e0; font-size: 16px;">Assistant: {assistant_response}</p>', unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": f"Assistant: {assistant_response}"})

            # Check if button clicked to convert speech
            if not st.session_state.is_button_clicked:
                button_clicked = st.button("Convert to Speech", on_click=on_voice)
                if button_clicked:
                    st.session_state.is_button_clicked = True
                    st.session_state.is_processing = True
                    time.sleep(2)
                    mp3_path = self.on_voice(assistant_response)
                    if mp3_path:
                        st.session_state.mp3_url = mp3_path
                    st.session_state.is_processing = False

        if st.session_state.is_processing:
            st.spinner("Processing...")

        # Sidebar to display MP3 files from the folder
        with st.sidebar:
            mp3_folder = "mp3"  # Define your MP3 folder
            mp3_files = [f for f in os.listdir(mp3_folder) if f.endswith(".mp3")]
            
            if mp3_files:
                # Display all available MP3 files in the folder
                selected_mp3 = st.selectbox("Select an MP3 file to play:", mp3_files)
                if selected_mp3:
                    # Generate the full path of the selected MP3 file
                    st.session_state.mp3_url = os.path.join(mp3_folder, selected_mp3)

            # Audio player
            if st.session_state.mp3_url:
                st.audio(st.session_state.mp3_url, format="audio/mp3")
            else:
                st.write("No audio available. Please convert a message to speech.")

if __name__ == "__main__":
    chat_ui = ChatUI()  
    chat_ui.render()

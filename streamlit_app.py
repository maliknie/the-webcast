import streamlit as st
import time

def handle_send(prompt):
    # Just echoes the prompt for demonstration.
    time.sleep(2)
    return f"This is a placeholder response to: {prompt}"

class ChatUI:
    def __init__(self, on_send=handle_send):
        """
        ChatUI constructor.

        Args:
            on_send (callable): Function that takes a prompt string and returns the assistant's response string.
        """
        self.on_send = on_send
        self._init_session_state()

    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render(self):
        st.title("WebCast")

        # Reset button
        if st.button("Reset Messages"):
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Type something..."):
            # Show user message
            with st.chat_message("user"):
                st.markdown(f"User: {prompt}")
            st.session_state.messages.append({"role": "user", "content": f"User: {prompt}"})

            with st.spinner('Assistant is typing...'):
                assistant_response = self.on_send(prompt)

            # Show assistant message
            with st.chat_message("assistant"):
                st.markdown(f"Assistant: {assistant_response}")
            st.session_state.messages.append({"role": "assistant", "content": f"Assistant: {assistant_response}"})




if __name__ == "__main__":
    chat_ui = ChatUI()
    chat_ui.render()
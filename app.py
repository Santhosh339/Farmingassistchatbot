import streamlit as st
from streamlit_chat import message
from model_tokeniser import model, tokenizer
from generate_responses import generate_response
from TextToSpeech import text_to_speech
from AudioToText import Speech_to_text

# Initialize chat history in session state if not present
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input section
user_input = st.text_input("You: ", key="input")

if user_input:
    # Add user input to chat history
    st.session_state['history'].append({"message": user_input, "is_user": True})
    
    # Generate bot response
    bot_response = generate_response(user_input, model, tokenizer)
    
    # Post-process bot response for cleaner output
    bot_response = bot_response.strip()  # Ensure no extra spaces
    st.session_state['history'].append({"message": bot_response, "is_user": False})

    # Optional: Add Text-to-Speech functionality
    try:
        if st.button("🎧", key="audio_button"):
            st.audio(text_to_speech(bot_response))
    except Exception as e:
        st.write(f"Error occurred: {e}")

# Display the chat history
for i, chat in enumerate(st.session_state['history']):
    message(chat['message'], is_user=chat['is_user'], key=f"message_{i}")

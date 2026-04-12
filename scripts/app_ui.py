import streamlit as st
import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000/generate"

st.set_page_config(
    page_title="Bigram LLM Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("Bigram LLM Assistant 🤖")
st.markdown("Chat with your custom trained language model!")

# Sidebar for generation parameters
with st.sidebar:
    st.header("Model Parameters")
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=1000, value=200, step=10,
                          help="Maximum number of tokens to generate.")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                           help="Higher values make output more random, lower values more deterministic.")
    top_k = st.slider("Top K", min_value=0, max_value=100, value=0, step=1,
                      help="Limits sampling to the top-K most likely tokens (0 to disable).")
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=1.0, step=0.05,
                      help="Nucleus sampling threshold (1.0 to disable).")
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.0, step=0.05,
                                  help="Penalize repeated tokens (1.0 means no penalty).")

    st.markdown("---")
    st.markdown("Status: Wait for the API server to be running before sending messages.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Build the payload
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty
        }
        
        try:
            with st.spinner("Generating..."):
                response = requests.post(API_URL, json=payload, timeout=60)
                
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("generated_text", "")
                
                # Streamlit doesn't support token-by-token streaming easily without Server-Sent Events from FastAPI
                # So we'll simulate a typing effect for aesthetic appeal
                full_response = ""
                # We slice the text into words/chunks rather than chars to be faster but still look like streaming
                for chunk in generated_text.split(" "):
                    full_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                    
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the API server. Make sure `uvicorn api:app` is running on port 8000.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

import streamlit as st
import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000/generate"
STREAM_URL = "http://localhost:8000/generate_stream"
AGENT_URL = "http://localhost:8000/agent_chat"
INGEST_URL = "http://localhost:8000/ingest"

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
    st.header("Advanced Features")
    use_agent = st.toggle("Enable Agent Mode", value=False, help="Allow model to use tools like Calculator and Wikipedia")
    use_stream = st.toggle("Enable Real Streaming", value=True, help="Stream response directly from server")
    use_speculative_decoding = st.toggle("Enable Speculative Decoding", value=False, help="Use a smaller draft model to speed up generation")
    use_rag = st.toggle("Enable RAG Context", value=False, help="Retrieve context from documents before generation")
    
    if use_rag:
        st.markdown("### Document Ingestion")
        doc_dir = st.text_input("Document Directory", value="data/documents", help="Path to directory containing .txt or .md files")
        if st.button("Ingest Documents"):
            try:
                with st.spinner("Ingesting..."):
                    res = requests.post(INGEST_URL, json={"doc_dir": doc_dir})
                if res.status_code == 200:
                    st.success(f"Ingested {res.json().get('chunks_ingested', 0)} chunks!")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

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
            "repetition_penalty": repetition_penalty,
            "use_speculative_decoding": use_speculative_decoding,
            "use_rag": use_rag
        }
        
        try:
            if use_agent:
                with st.spinner("Agent is thinking..."):
                    response = requests.post(AGENT_URL, json={"prompt": prompt}, timeout=120)
                if response.status_code == 200:
                    full_response = response.json().get("response", "")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            elif use_stream:
                response = requests.post(STREAM_URL, json=payload, stream=True, timeout=60)
                if response.status_code == 200:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                data = decoded_line[6:]
                                if data == "[DONE]":
                                    break
                                full_response += data
                                message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            else:
                with st.spinner("Generating..."):
                    response = requests.post(API_URL, json=payload, timeout=60)
                    
                if response.status_code == 200:
                    data = response.json()
                    generated_text = data.get("generated_text", "")
                    
                    full_response = ""
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

import streamlit as st
import requests
import pandas as pd
import re
from io import StringIO
import os

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"

# --- Page Setup ---
st.set_page_config(
    page_title="AutoML Agent Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

# --- Helper Functions ---
def parse_assistant_response(response_text):
    """
    Parses the assistant's response to extract special components
    like DataFrames or image paths.
    """
    plot_path = None
    dataframe = None
    
    plot_path_match = re.search(r"`(plots/[^`]+\.png)`", response_text)
    if plot_path_match and os.path.exists(plot_path_match.group(1)):
        plot_path = plot_path_match.group(1)
        response_text = re.sub(r"`(plots/[^`]+\.png)`", "", response_text)

    dataframe_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
    if dataframe_match:
        df_string = dataframe_match.group(1)
        try:
            # Use pandas to read the string as if it were a file
            # This regex helps handle the index column that pandas.to_string() creates
            df_string_cleaned = re.sub(r'^\s*\d+\s', '', df_string, flags=re.MULTILINE)
            df = pd.read_csv(StringIO(df_string_cleaned), sep='\s{2,}', engine='python')
            dataframe = df
            response_text = re.sub(r"```.*```", "\n*Result displayed below.*", response_text, flags=re.DOTALL)
        except Exception as e:
            st.warning(f"Could not parse DataFrame from text: {e}")
            pass
            
    return response_text.strip(), dataframe, plot_path

def start_new_chat():
    """Resets the session state to start a new conversation."""
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.chat_active = False

# --- Sidebar for Session Control ---
with st.sidebar:
    st.title("AutoML Agent")
    st.markdown("---")
    
    if st.button("➕ New Chat"):
        start_new_chat()
        st.rerun()

    st.markdown("### 1. Start a New Session")
    uploaded_file = st.file_uploader(
        "Upload your CSV Dataset", 
        type="csv",
        on_change=start_new_chat 
    )

# --- Main Chat Interface ---
st.header("🤖 AutoML Chatbot")

# Handle file upload to start a session
if uploaded_file is not None and not st.session_state.chat_active:
    with st.spinner("Uploading and processing dataset..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_URL}/upload", files=files)
            response.raise_for_status()
            
            data = response.json()
            st.session_state.session_id = data["session_id"]
            
            # --- KEY FIX ---
            # Manually construct the full message with the data preview
            # wrapped in markdown code blocks so our parser can find it.
            full_welcome_message = f"{data['message']}\n\n**Data Preview:**\n```\n{data['data_preview']}\n```"
            
            text_res, df_res, _ = parse_assistant_response(full_welcome_message)
            
            initial_message = {"role": "assistant", "content": text_res}
            if df_res is not None:
                initial_message["dataframe"] = df_res
                
            st.session_state.messages.append(initial_message)
            st.session_state.chat_active = True
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading file: {e}")

# Display the entire chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("dataframe") is not None:
            st.dataframe(message["dataframe"])
        if message.get("plot_path") is not None:
            st.image(message["plot_path"])

# Get new user input
if st.session_state.chat_active:
    if prompt := st.chat_input("Describe your ML goal or next step..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"session_id": st.session_state.session_id, "user_query": prompt}
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    text_res, df_res, plot_res = parse_assistant_response(data["assistant_message"])
                    
                    assistant_response = {"role": "assistant", "content": text_res}
                    if df_res is not None:
                        assistant_response["dataframe"] = df_res
                    if plot_res is not None:
                        assistant_response["plot_path"] = plot_res
                        
                    st.session_state.messages.append(assistant_response)
                    st.rerun()

                except requests.exceptions.RequestException as e:
                    error_message = f"Error connecting to the backend: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Please upload a CSV file in the sidebar to start a new chat session.")

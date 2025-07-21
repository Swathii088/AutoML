import streamlit as st
import requests
import pandas as pd
import re
from io import StringIO

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/chat"
DATASETS_URL = "http://127.0.0.1:8000/datasets"

# --- Page Setup ---
st.set_page_config(
    page_title="AutoML Agent Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AutoML Agent Chatbot")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initial_datasets_loaded" not in st.session_state:
    st.session_state.initial_datasets_loaded = False
# NEW: State for managing views (chat vs. plot)
if "view" not in st.session_state:
    st.session_state.view = "chat"
if "plot_to_show" not in st.session_state:
    st.session_state.plot_to_show = None

# --- Helper Functions ---
def get_initial_datasets():
    """Fetches the list of available datasets from the API."""
    try:
        response = requests.get(DATASETS_URL)
        response.raise_for_status()
        datasets = response.json()
        formatted_list = "\n".join(datasets)
        return f"Welcome! To get started, please load a dataset from the following list:\n\n{formatted_list}"
    except requests.exceptions.RequestException:
        return f"Error: Could not connect to the backend API at {API_URL}. Please ensure the FastAPI server is running."

def handle_assistant_response(response_text):
    """Parses the assistant's response for special content like plots or dataframes."""
    content_to_display = {"role": "assistant"}
    
    plot_path_match = re.search(r"`(plots/[^`]+\.png)`", response_text)
    dataframe_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)

    if plot_path_match:
        path = plot_path_match.group(1)
        # Instead of displaying, we store the path and create a button later
        content_to_display["plot_path"] = path
        response_text = re.sub(r"`(plots/[^`]+\.png)`", "", response_text)

    if dataframe_match:
        df_string = dataframe_match.group(1)
        try:
            df = pd.read_csv(StringIO(df_string), sep='\s{2,}', engine='python')
            content_to_display["dataframe"] = df.to_json()
            response_text = re.sub(r"```.*```", "(DataFrame displayed below)", response_text, flags=re.DOTALL)
        except Exception:
            pass
            
    content_to_display["content"] = response_text.strip()
    return content_to_display

def show_plot_view():
    """Displays the dedicated view for a single plot."""
    st.header("Plot Viewer")
    
    if st.button("⬅️ Back to Chat"):
        st.session_state.view = "chat"
        st.rerun()

    if st.session_state.plot_to_show:
        try:
            st.image(st.session_state.plot_to_show, use_column_width=True)
        except Exception as e:
            st.error(f"Could not display image at {st.session_state.plot_to_show}: {e}")
    else:
        st.warning("No plot to display.")

def show_chat_view():
    """Displays the main chat interface."""
    # Display the chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if "dataframe" in message:
                st.dataframe(pd.read_json(message["dataframe"]))
            
            # NEW: If a message has a plot path, show a button
            if "plot_path" in message:
                if st.button(f"View Plot: {message['plot_path']}", key=f"plot_btn_{i}"):
                    st.session_state.plot_to_show = message["plot_path"]
                    st.session_state.view = "plot"
                    st.rerun()
            
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("What would you like to do?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "user_query": prompt
                }
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                
                data = response.json()
                st.session_state.session_id = data["session_id"]
                
                assistant_message = handle_assistant_response(data["assistant_message"])
                st.session_state.messages.append(assistant_message)
                
                st.rerun()

            except requests.exceptions.RequestException as e:
                error_message = f"Error connecting to the backend: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Main App Logic ---

# Load initial datasets message only once
if not st.session_state.initial_datasets_loaded:
    welcome_message = get_initial_datasets()
    if "Error" not in welcome_message:
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        st.session_state.initial_datasets_loaded = True
    else:
        st.error(welcome_message)


# Router to show either the chat or the plot view
if st.session_state.view == "plot":
    show_plot_view()
else:
    show_chat_view()

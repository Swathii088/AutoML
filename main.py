import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uuid
import pandas as pd

# --- Import your project's core components ---
from state.graph_state import MLState
from agent.router import get_routing_decision
from tools.load_dataset_tool import load_dataset_tool, list_available_datasets
from llm.task_identifier_nodes import (
    identify_task_node,
    request_confirmation_node,
    handle_validation_node,
)

# --- Import All Tool Logic Functions ---
from tools.setup_tool import setup_tool
from tools.compare_models_tool import compare_models_tool
from tools.create_model_tool import  create_model_tool
from tools.tune_model_tool import  tune_model_tool
from tools.ensemble_model_tool import ensemble_model_tool
from tools.automl_tool import automl_tool
from tools.leaderboard_tool import leaderboard_tool
from tools.finalize_model_tool import finalize_model_tool
from tools.save_model_tool import save_model_tool
from tools.load_model_tool import load_model_tool
from tools.assign_model_tool import assign_model_tool_logic
from tools.data_analysis_tool import (
    descriptive_statistics_tool,
    missing_values_tool,
    correlation_analysis_tool,
)
from tools.plot_tool import plot_model_tool

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="AutoML Agent API",
    description="An API for interacting with a natural-language-driven ML system.",
)

# --- In-Memory Session Storage ---
# For local testing, a simple dictionary is perfect to store session states.
sessions: Dict[str, MLState] = {}

# --- Pydantic Models for API Requests/Responses ---
class ChatRequest(BaseModel):
    session_id: str | None = None
    user_query: str

class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str
    state_summary: Dict[str, Any] # A JSON-safe summary of the state

# --- Helper Function to Create a Serializable State Summary ---
def create_serializable_state_summary(state: MLState) -> Dict[str, Any]:
    """
    Creates a JSON-safe dictionary summary of the MLState, excluding complex objects.
    """
    summary = {
        "data_loaded": state.data is not None,
        "data_shape": state.data.shape if isinstance(state.data, pd.DataFrame) else None,
        "task": state.task,
        "target_column": state.target_column,
        "setup_done": state.setup_done,
        "model_available": any([
            state.model, state.tuned_model, state.final_model, 
            state.best_model, state.ensemble_model
        ]),
        "leaderboard_available": state.leaderboard is not None,
    }
    # Return only the keys that have a value
    return {k: v for k, v in summary.items() if v is not None}

# --- Tool Mapping ---
tool_map = {
    "load_dataset_tool": load_dataset_tool,
    "descriptive_statistics_tool": descriptive_statistics_tool,
    "missing_values_tool":  missing_values_tool,
    "correlation_analysis_tool": correlation_analysis_tool,
    "plot_model_tool": plot_model_tool,
    "setup_tool":  setup_tool,
    "compare_models_tool": compare_models_tool,
    "create_model_tool": create_model_tool,
    "tune_model_tool": tune_model_tool,
    "ensemble_model_tool": ensemble_model_tool,
    "automl_tool": automl_tool,
    "leaderboard_tool": leaderboard_tool,
    "finalize_model_tool": finalize_model_tool,
    "save_model_tool": save_model_tool,
    "load_model_tool": load_model_tool,
    "assign_model_tool": assign_model_tool_logic,
}

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the AutoML Agent API. Use the /chat endpoint to interact."}

@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    """
    Main endpoint for interacting with the AutoML agent.
    """
    session_id = request.session_id or str(uuid.uuid4())
    state = sessions.get(session_id, MLState(input_message=""))
    user_query = request.user_query
    
    # --- Onboarding & Main Logic ---
    if state.data is None:
        updates = load_dataset_tool.invoke({"state": state, "user_query": f"load {user_query}"})
        state = state.model_copy(update=updates)
    elif state.task is None:
        state.input_message = user_query
        updates = identify_task_node(state)
        state = state.model_copy(update=updates)
        updates = request_confirmation_node(state)
        state = state.model_copy(update=updates)
    elif not state.setup_done and state.task and state.user_validation_response is None:
         state.user_validation_response = user_query
         updates = handle_validation_node(state)
         state = state.model_copy(update=updates)
    else:
        decision = get_routing_decision(user_query)
        if not decision or "tool_name" not in decision:
            state.last_output = "I'm sorry, I couldn't understand that command."
        else:
            tool_name = decision["tool_name"]
            tool_to_call = tool_map.get(tool_name)
            if not tool_to_call:
                state.last_output = f"Tool '{tool_name}' is not implemented."
            else:
                updates = tool_to_call.invoke({"state": state, "user_query": user_query})
                state = state.model_copy(update=updates)

    # Save the updated state for the session
    sessions[session_id] = state
    
    # Return the serializable summary
    return ChatResponse(
        session_id=session_id,
        assistant_message=state.last_output,
        state_summary=create_serializable_state_summary(state)
    )

@app.get("/datasets")
def get_datasets() -> List[str]:
    """Endpoint to get the list of available datasets."""
    # We call the underlying function, not the tool
    return list_available_datasets().split('\n')

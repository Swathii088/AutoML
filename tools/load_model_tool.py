import logging
import os
import json
from typing import Dict, Any, Optional
from pycaret.classification import load_model as classification_load
from pycaret.regression import load_model as regression_load
from pycaret.clustering import load_model as clustering_load
from pycaret.anomaly import load_model as anomaly_load
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState 
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def _get_llm_load_parameters(user_query: str) -> Dict[str, Any] | None:
    """
    Uses an LLM to extract the filename to load.
    Expected JSON output: {"filename": "extracted_name"}
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are an assistant that extracts information. The user wants to load a model.
Extract the desired filename from their request. The filename is a string with no extension.
You must respond with a valid JSON object in the format: {"filename": "extracted_name"}.
If no valid filename is found, return {"filename": null}.
"""),
        ("user", "User Query: {user_query}")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({"user_query": user_query})
        return json.loads(response.content.strip())
    except json.JSONDecodeError as e:
        logger.error(f"LLM did not return a valid JSON object for load_model parameters: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM call for load_model parameters failed: {e}")
        return None

@tool("load_model_tool", return_direct=True) # Added return_direct=True
def load_model_tool(state: MLState, user_query: str) -> Dict[str, Any]:
    """
    Loads a previously saved model pipeline from a file into the state.

    This tool requires the user to specify the filename of the model to load.
    The loaded model becomes the active model in the state, ready for prediction.

    Args:
        state (MLState): The current state of the ML pipeline. The task type
                         ('classification', 'regression', 'clustering', 'anomaly') must be set.
        user_query (str): The user's command, which must include the filename of the
                          model to load (e.g., "load the model my_classifier").

    Returns:
        Dict[str, Any]: A dictionary containing the loaded `model` object and other
                        relevant state updates.
    """
    logger.info("--- Executing Load Model Tool ---")
    
    task = state.task
    if not task:
        # A task must be set to know which PyCaret function to use
        return {"last_output": "❌ Please specify the task type (e.g., 'classification', 'regression', 'clustering', 'anomaly') before loading a model."}

    llm_params = _get_llm_load_parameters(user_query)
    if not llm_params or llm_params.get("filename") is None:
        return {"last_output": "❌ Please specify the filename of the model you want to load. Example: 'load the model my_best_pipeline'."}
    
    filename = llm_params["filename"]
    
    # Ensure the 'models' directory exists
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"Created directory: {model_dir}")

    model_full_path = os.path.join(model_dir, f"{filename}.pkl")

    if not os.path.exists(model_full_path):
        return {"last_output": f"❌ Model file not found at '{model_full_path}'. Please ensure the model exists and the filename is correct."}

    load_map = {
        "classification": classification_load,
        "regression": regression_load,
        "clustering": clustering_load,
        "anomaly": anomaly_load
    }
    
    if task not in load_map:
        return {"last_output": f"❌ Loading models for task type '{task}' is not supported. Supported tasks: {', '.join(load_map.keys())}"}

    try:
        load_function = load_map[task]
        loaded_model = load_function(model_full_path)
        
        logger.info(f"Model '{filename}' loaded successfully for task '{task}'.")
        return {
            "model": loaded_model,
            "final_model": loaded_model, # A loaded model is considered final for immediate use
            "model_path": model_full_path,
            "model_name": filename,
            "last_output": f"✅ Model '{filename}' loaded successfully and is now the active model for {task} tasks."
        }
    except Exception as e:
        logger.error(f"Failed to load model '{filename}': {e}", exc_info=True)
        return {"last_output": f"❌ Failed to load model '{filename}': {e}. Please ensure the model file is not corrupted and matches the current PyCaret version and task type."}


import logging
import os
import json
from typing import Dict, Any, Optional
from pycaret.classification import save_model as classification_save
from pycaret.regression import save_model as regression_save
from pycaret.clustering import save_model as clustering_save
from pycaret.anomaly import save_model as anomaly_save
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState 
from dotenv import load_dotenv

load_dotenv()

# Configure logging and LLM
logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def _get_llm_save_parameters(user_query: str) -> Dict[str, Any] | None:
    """
    Uses an LLM to extract a filename from the user query.
    Expected JSON output: {"filename": "extracted_name"}
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are an assistant that extracts information. The user wants to save a model.
Extract the desired filename from their request. The filename should be a string with no extension.
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
        logger.error(f"LLM did not return a valid JSON object for save_model parameters: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM call for save_model parameters failed: {e}")
        return None

@tool("save_model_tool", return_direct=True) 
def save_model_tool(state: MLState, user_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Saves the best available trained model to a file.

    This tool intelligently selects the most recently finalized or improved model
    from the state and saves it to the './models/' directory. A custom filename
    can be provided in the user query.

    Args:
        state (MLState): The current state of the ML pipeline.
        user_query (Optional[str]): The user's command, which may include a
                                     custom filename for the model.

    Returns:
        Dict[str, Any]: A dictionary containing the `model_path` and `last_output`.
    """
    logger.info("--- Executing Save Model Tool ---")

    model_to_save = (
        state.final_model or
        state.ensemble_model or
        state.tuned_model or
        state.best_model or
        state.created_model
    )

    if isinstance(model_to_save, list) and len(model_to_save) > 0:

        model_to_save = model_to_save[0]
        logger.info(f"Multiple models found in state. Selecting the first one for saving: {type(model_to_save).__name__}")
    elif model_to_save is None:
        return {"last_output": "❌ No trained model is available to save. Please create, compare, tune, or finalize a model first."}

    task = state.task
    if not task:
        return {"last_output": "❌ Task type is not set in the state. Cannot determine which PyCaret module's save_model to use."}
    
    #
    save_map = {
        "classification": classification_save,
        "regression": regression_save,
        "clustering": clustering_save,
        "anomaly": anomaly_save
    }
    
    if task not in save_map:
        return {"last_output": f"❌ `save_model` tool is not applicable for the '{task}' task. Supported tasks: {', '.join(save_map.keys())}"}

    filename = "final_model" 
    if user_query:
        logger.info(f"Parsing user query for filename: '{user_query}'")
        llm_params = _get_llm_save_parameters(user_query)
        if llm_params and llm_params.get("filename") is not None: 
            filename = llm_params["filename"]
            logger.info(f"Using custom filename from user: '{filename}'")

    try:
       
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True) 
        save_path = os.path.join(model_dir, filename)
        
        save_function = save_map[task]
        save_function(model_to_save, save_path)
        
        full_path = f"{save_path}.pkl"
        logger.info(f"Model saved successfully to {full_path}")
        
        model_details = str(model_to_save) 
        
        return {
            "model_path": full_path,
            "model_name": filename, # Store the name used
            "last_output": f"✅ Model saved successfully to `{full_path}`.\n\n**Model Details:**\n```\n{model_details}\n```"
        }

    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)
        return {"last_output": f"❌ Failed to save model: {e}. Please ensure the model is valid and the path is writable."}


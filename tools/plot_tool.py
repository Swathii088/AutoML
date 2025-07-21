import logging
import os
import json
from typing import Dict, Any, Optional
from pycaret.classification import plot_model as classification_plot
from pycaret.regression import plot_model as regression_plot
from pycaret.clustering import plot_model as clustering_plot
from pycaret.anomaly import plot_model as anomaly_plot
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

# --- Configuration ---

# A comprehensive list of valid plot types for each task
VALID_PLOTS = {
    "classification": ['auc', 'confusion_matrix', 'feature', 'pr', 'class_report', 'learning', 'boundary', 'calibration', 'dimension', 'error', 'ks', 'lift', 'mcc', 'rfe', 'roc', 'threshold'],
    "regression": ['residuals', 'error', 'cooks', 'learning', 'feature', 'manifold', 'rfe', 'residuals_interactive'],
    "clustering": ['cluster', 'tsne', 'elbow', 'silhouette', 'distance', 'distribution'],
    "anomaly": ['tsne', 'umap']
}

def _get_active_model(state: MLState) -> Any | None:
    """Intelligently selects the best available model from the state."""
    model = (
        state.ensemble_model or state.tuned_model or state.best_model
        or state.created_model or state.final_model
    )
    if isinstance(model, list): return model[0]
    return model

def _get_llm_plot_parameters(user_query: str, task: str) -> Dict[str, Any] | None:
    """Uses an LLM to extract the plot type from the user query."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that extracts the `plot` type for PyCaret's `plot_model` function from a user query. You must respond with a valid JSON object."),
        ("user", "Valid plot types for a '{task}' task are: {plot_options}\n\nUser Query: {user_query}\n\nJSON Output (e.g., {{\"plot\": \"feature\"}}):")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "task": task,
            "plot_options": VALID_PLOTS.get(task, []),
            "user_query": user_query
        })
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"LLM call for plot parameters failed: {e}")
        return None

@tool("plot_model_tool")
def plot_model_tool(state: MLState, user_query: str) -> Dict[str, Any]:
    """
    Generates and saves a plot for the current model.

    This tool uses an LLM to identify the desired plot type (e.g., 'feature importance',
    'confusion matrix') from the user's query, generates the plot, saves it as a
    PNG file, and returns the file path.

    Args:
        state (MLState): The current state of the ML pipeline.
        user_query (str): The user's command specifying which plot to create.

    Returns:
        Dict[str, Any]: A dictionary containing the `plot_path` and `last_output`.
    """
    logger.info("--- Executing Plot Model Tool ---")
    
    model = _get_active_model(state)
    if not model:
        return {"last_output": "❌ No trained model found. Please train a model first."}

    task = state.task
    if task not in VALID_PLOTS:
        return {"last_output": f"❌ Plotting is not supported for the '{task}' task."}

    params = _get_llm_plot_parameters(user_query, task)
    if not params or not params.get("plot"):
        return {"last_output": "❌ I could not identify which plot you want to create."}
    
    plot_type = params["plot"]
    if plot_type not in VALID_PLOTS[task]:
        return {"last_output": f"❌ '{plot_type}' is not a valid plot for a {task} task."}

    plot_map = {
        "classification": classification_plot, "regression": regression_plot,
        "clustering": clustering_plot, "anomaly": anomaly_plot
    }

    try:
        os.makedirs("plots", exist_ok=True)
        plot_function = plot_map[task]
        
        # PyCaret's plot_model returns the file path when save=True
        file_path = plot_function(model, plot=plot_type, save=True, verbose=False)
        
        logger.info(f"Successfully generated plot and saved to '{file_path}'")
        
        return {
            "plot_path": file_path,
            "last_output": f"✅ Plot '{plot_type}' generated successfully. You can view it at: `{file_path}`"
        }
    except Exception as e:
        logger.error(f"Failed to generate plot '{plot_type}': {e}", exc_info=True)
        return {"last_output": f"❌ Failed to generate plot '{plot_type}': {e}"}
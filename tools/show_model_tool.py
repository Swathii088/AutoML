import logging
from typing import Dict, Any

from langchain_core.tools import tool
from state.graph_state import MLState # Adjust import path as needed

logger = logging.getLogger(__name__)

@tool("show_model_tool")
def show_model_tool(state: MLState, user_query: str = None) -> Dict[str, Any]:
    """
    Displays the parameters of the best available model in the current state.
    
    This tool intelligently selects the most advanced model from the state 
    (finalized > ensembled > tuned > best > created) and shows its details.
    
    Args:
        state (MLState): The current state of the ML pipeline.
        user_query (str): The user's query (not used in this tool but required for consistency).

    Returns:
        Dict[str, Any]: A dictionary containing the `last_output` with the
                        model's details to update the state.
    """
    logger.info("--- Executing Show Model Tool ---")

    # CORRECTED LOGIC: Prioritize the generic 'model' field, which always
    # holds the most recently created/updated model. The others are fallbacks.
    model_to_show = (
        state.model or
        state.final_model or
        state.ensemble_model or
        state.tuned_model or
        state.best_model or
        state.created_model
    )

    if model_to_show is None:
        return {"last_output": "❌ No trained model is available to show."}

    # If the selected model is a list (from compare_models), pick the first one
    if isinstance(model_to_show, list):
        model_to_show = model_to_show[0]

    try:
        model_details = str(model_to_show)
        logger.info(f"Displaying details for model: {type(model_to_show).__name__}")

        return {
            "last_output": f"**Current Active Model:**\n```\n{model_details}\n```"
        }
    except Exception as e:
        logger.error(f"Failed to show model: {e}", exc_info=True)
        return {"last_output": f"❌ Failed to display model details: {e}"}

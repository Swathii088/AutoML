import logging
import pandas as pd 
from typing import Dict, Any
from langchain_core.tools import tool
from state.graph_state import MLState 

logger = logging.getLogger(__name__)

@tool("leaderboard_tool", return_direct=True)
def leaderboard_tool(state: MLState) -> Dict[str, Any]:
    """
    Retrieves and displays the model comparison leaderboard from the state.

    This tool should be used after `compare_models` or `automl` has been run.
    It formats the leaderboard DataFrame for display to the user.

    Args:
        state (MLState): The current state of the ML pipeline.

    Returns:
        Dict[str, Any]: A dictionary containing the `last_output` with the
                        formatted leaderboard to update the state.
    """
    logger.info("--- Executing Leaderboard Tool ---")
    
    leaderboard = state.leaderboard

    # Check if leaderboard is None or an empty DataFrame
    if leaderboard is None or (isinstance(leaderboard, pd.DataFrame) and leaderboard.empty):
        logger.warning("Attempted to access leaderboard, but none was found or it was empty.")
        return {"last_output": "❌ No leaderboard is available. Please run `compare_models` or `automl` first to generate a leaderboard."}

    try:
        # Format the leaderboard DataFrame into a string for display
        leaderboard_str = leaderboard.to_string()
        logger.info("Successfully retrieved and formatted the leaderboard.")
        
        return {
            "last_output": f"**📊 Model Comparison Leaderboard:**\n```\n{leaderboard_str}\n```"
        }
    except Exception as e:
        logger.error(f"Failed to display leaderboard: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred while trying to display the leaderboard: {e}"}
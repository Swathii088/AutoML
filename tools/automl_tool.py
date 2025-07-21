import logging
from typing import Dict, Any
from pycaret.classification import compare_models as classification_compare, pull as classification_pull
from pycaret.regression import compare_models as regression_compare, pull as regression_pull
from langchain_core.tools import tool
from state.graph_state import MLState 

logger = logging.getLogger(__name__)

@tool("automl_tool")
def automl_tool(state: MLState) -> Dict[str, Any]:
    """
    Runs a full AutoML process by comparing all baseline models with default settings.

    This tool is a simplified entrypoint to model comparison for users who do not
    want to specify any custom parameters. It selects the best overall model and
    stores it and the leaderboard in the state.

    Args:
        state (MLState): The current state of the ML pipeline. Must have a
                         completed setup.

    Returns:
        Dict[str, Any]: A dictionary containing the `best_model`, `leaderboard`,
                        and `last_output` to update the state.
    """
    logger.info("--- Executing AutoML Tool ---")

    if not state.setup_done:
        return {"last_output": "❌ Please run the setup tool before using AutoML."}

    task = state.task
    if task not in ("classification", "regression"):
        return {"last_output": f"❌ AutoML is not applicable for the '{task}' task."}

    logger.info(f"Running AutoML for {task} task with default settings...")

    compare_map = {
        "classification": classification_compare,
        "regression": regression_compare
    }
    pull_map = {
        "classification": classification_pull,
        "regression": regression_pull
    }

    try:
        compare_function = compare_map[task]
        best_model = compare_function(verbose=False, n_select=1)
        leaderboard = pull_map[task]()
        
        logger.info("AutoML process completed successfully.")
        
        return {
            "best_model": best_model,
            "leaderboard": leaderboard,
            "last_output": f"✅ AutoML complete. The best model has been selected.\n\n**Leaderboard:**\n```\n{leaderboard.to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"AutoML process failed: {e}", exc_info=True)
        return {"last_output": f"❌ AutoML failed: {e}"}
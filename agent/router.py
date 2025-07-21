import logging
import json
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)
ROUTER_PROMPT_TEMPLATE = """
You are an expert at routing a user's request to the correct tool.
Based on the user's query, you must select the single best tool to call from the list of available tools.
You must respond with a JSON object with the key "tool_name" and the user's original query as "user_query".

Available Tools:
- `load_dataset_tool`: Use when the user wants to load a dataset (e.g., "load boston").
- `descriptive_statistics_tool`: Use for summary statistics (e.g., "describe the data").
- `missing_values_tool`: Use when the user asks about missing or null values.
- `correlation_analysis_tool`: Use for correlation analysis.
- `plot_model_tool`: Use when the user asks to plot or visualize something.
- `setup_tool`: Use to initialize the ML environment (e.g., "run setup with normalization").
- `compare_models_tool`: Use to compare and rank models.
- `create_model_tool`: Use to train a single, specific model (e.g., "create a random forest").
- `tune_model_tool`: Use to tune a model's hyperparameters.
- `ensemble_model_tool`: Use to create an ensemble model.
- `automl_tool`: Use for a fully automatic model search.
- `leaderboard_tool`: Use to display the model leaderboard.
- `finalize_model_tool`: Use to finalize a model before saving.
- `save_model_tool`: Use to save a model.
- `load_model_tool`: Use to load a saved model.

User Query: {user_query}

JSON Output:
"""

def get_routing_decision(user_query: str) -> Dict[str, Any] | None:
    """
    Uses the LLM to decide which tool to call based on the user's query.

    Args:
        user_query (str): The user's command.

    Returns:
        A dictionary with "tool_name" and "user_query", or None if it fails.
    """
    logger.info(f"Routing query: '{user_query}'")
    
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"user_query": user_query})
        decision = json.loads(response.content)
        logger.info(f"Router decision: {decision}")
        return decision
    except Exception as e:
        logger.error(f"Router LLM call failed: {e}")
        return None


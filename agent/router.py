import logging
import json
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# --- Configure logging and LLM ---
logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192", # Using a powerful model for routing is key
    model_kwargs={"response_format": {"type": "json_object"}},
)

# --- OPTIMIZED ROUTER PROMPT ---
ROUTER_PROMPT_TEMPLATE = """
You are an expert at routing a user's natural language query to the single most appropriate tool. Your only job is to analyze the query and select one tool from the list below. You must respond with a JSON object containing the chosen "tool_name" and the original "user_query".

Based on the user's intent, select one of the following tools:

**1. Data Loading & Setup:**
- `load_dataset_tool`: When the user wants to load a new dataset. (e.g., "load boston", "get the iris data")

**2. Data Exploration (EDA):**
- `descriptive_statistics_tool`: For summary statistics. (e.g., "describe the data", "give me some stats")
- `missing_values_tool`: To check for missing or null values. (e.g., "any missing values?", "check for nulls")
- `correlation_analysis_tool`: For correlation analysis. (e.g., "show me correlations", "are any features correlated?")

**3. Data Preprocessing:**
- `setup_tool`: To initialize the ML environment and apply preprocessing. (e.g., "run setup with normalization", "preprocess the data for me", "let's get the data ready")

**4. Modeling & AutoML:**
- `compare_models_tool`: To compare and rank multiple models. (e.g., "compare models", "find the best algorithm")
- `automl_tool`: For a fully automatic model search without specifics. (e.g., "run automl", "find the best model automatically")
- `create_model_tool`: To train a single, specific model. (e.g., "create a random forest", "train a kmeans model")
- `tune_model_tool`: To tune a model's hyperparameters. (e.g., "tune the model", "optimize the random forest")
- `ensemble_model_tool`: To create an ensemble model. (e.g., "ensemble the best model", "try bagging")

**5. Model Inspection & Results:**
- `leaderboard_tool`: To display the model comparison leaderboard. (e.g., "show the leaderboard")
- `show_model_tool`: To display the parameters of the current active model. (e.g., "show me the model")
- `assign_model_tool`: For unsupervised tasks, to see results on the training data. (e.g., "assign the clusters", "show me the anomalies")

**6. Visualization:**
- `plot_model_tool`: When the user asks to plot or visualize something related to a model. (e.g., "plot feature importance", "show me the confusion matrix")

**7. Model Persistence:**
- `finalize_model_tool`: To finalize a model before saving. (e.g., "finalize the model")
- `save_model_tool`: To save a model. (e.g., "save the model as my_predictor")
- `load_model_tool`: To load a saved model. (e.g., "load my_predictor")

---
User Query: {user_query}

JSON Output:
"""

# --- The Router Function ---
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
        
        # Basic validation
        if "tool_name" not in decision:
            logger.warning(f"Router LLM failed to return 'tool_name'. Response: {decision}")
            return None

        logger.info(f"Router decision: {decision['tool_name']}")
        decision['user_query'] = user_query # Ensure the original query is included
        return decision
        
    except Exception as e:
        logger.error(f"Router LLM call failed: {e}")
        return None
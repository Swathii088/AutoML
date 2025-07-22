import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

from langchain_core.tools import tool
from state.graph_state import MLState

# Import the specific pull functions from each PyCaret module
from pycaret.classification import pull as classification_pull
from pycaret.regression import pull as regression_pull
from pycaret.clustering import pull as clustering_pull
from pycaret.anomaly import pull as anomaly_pull

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- NEW: Helper function to get the correct data (raw vs. transformed) ---
def _get_active_dataframe(state: MLState) -> pd.DataFrame | None:
    """
    Returns the transformed dataframe if setup is complete, otherwise returns
    the original raw dataframe.
    """
    if state.setup_done:
        logger.info("Setup is complete. Pulling transformed data for analysis.")
        pull_map = {
            "classification": classification_pull,
            "regression": regression_pull,
            "clustering": clustering_pull,
            "anomaly": anomaly_pull,
        }
        pull_function = pull_map.get(state.task)
        if pull_function:
            # The pull() function with no arguments returns the transformed dataset
            return pull_function()
    
    logger.info("Setup not complete. Using original raw data for analysis.")
    return state.data

# --- Tool 1: Descriptive Statistics (Updated) ---
@tool("descriptive_statistics_tool")
def descriptive_statistics_tool(state: MLState) -> Dict[str, Any]:
    """
    Calculates and displays descriptive statistics for the current dataset.
    If setup has been run, it analyzes the transformed data.
    """
    logger.info("--- Executing Descriptive Statistics Tool ---")
    
    data = _get_active_dataframe(state) # Use the helper function
    if data is None:
        return {"last_output": "❌ No data available for analysis."}

    try:
        numeric_summary = data.select_dtypes(include=np.number).describe().to_string()
        
        summary_str = f"### Data Shape\n{data.shape}\n\n"
        summary_str += f"### Numeric Summary\n```\n{numeric_summary}\n```\n\n"
        
        return { "last_output": f"**📊 Descriptive Statistics:**\n{summary_str}" }
        
    except Exception as e:
        logger.error(f"Failed to generate descriptive statistics: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred while generating statistics: {e}"}


# --- Tool 2: Missing Values Analysis (Updated) ---
@tool("missing_values_tool")
def missing_values_tool(state: MLState) -> Dict[str, Any]:
    """
    Analyzes and reports missing values for the current dataset.
    If setup has been run, it analyzes the transformed data.
    """
    logger.info("--- Executing Missing Values Tool ---")
    
    data = _get_active_dataframe(state) # Use the helper function
    if data is None:
        return {"last_output": "❌ No data available for analysis."}

    try:
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_counts,
            'Percentage (%)': missing_percentage
        })
        
        missing_df = missing_df[missing_df['Missing Values'] > 0]

        if missing_df.empty:
            return {"last_output": "✅ No missing values were found in the dataset."}
        
        return {
            "last_output": f"**🔍 Missing Values Analysis:**\n```\n{missing_df.to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"Failed to analyze missing values: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred while analyzing missing values: {e}"}


# --- Tool 3: Correlation Analysis (Updated) ---
@tool("correlation_analysis_tool")
def correlation_analysis_tool(state: MLState) -> Dict[str, Any]:
    """
    Computes the correlation matrix for the current dataset's numeric variables.
    If setup has been run, it analyzes the transformed data.
    """
    logger.info("--- Executing Correlation Analysis Tool ---")
    
    data = _get_active_dataframe(state) # Use the helper function
    if data is None:
        return {"last_output": "❌ No data available for analysis."}

    try:
        numeric_data = data.select_dtypes(include=np.number)
        if len(numeric_data.columns) < 2:
            return {"last_output": "Not enough numeric columns to compute correlations."}
            
        corr_matrix = numeric_data.corr()
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corrs.append(
                        f"- `{corr_matrix.columns[i]}` and `{corr_matrix.columns[j]}`: {corr_matrix.iloc[i, j]:.2f}"
                    )
        
        high_corr_summary = "\n".join(high_corrs) if high_corrs else "No highly correlated pairs found."
        
        full_report = f"""
### 🔗 High Correlation Pairs (>0.8)
{high_corr_summary}

---

### Full Correlation Matrix
```
{corr_matrix.to_string()}
```
"""
        return {"last_output": full_report}

    except Exception as e:
        logger.error(f"Failed to compute correlations: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred during correlation analysis: {e}"}

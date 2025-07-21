from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class MLState(BaseModel):
    """
    Represents the state of the machine learning pipeline.
    This model is used to manage and share data between agents in LangGraph.
    """
    
    # --- Core Data and Task Information ---
    data: Optional[Any] = Field(None, description="The loaded dataset, typically a pandas DataFrame.")
    task: Optional[str] = Field(None, description="ML task type: 'classification', 'regression', 'clustering', etc.")
    target_column: Optional[str] = Field(None, description="Target column for supervised learning. Is None for unsupervised tasks.")

    # Setup and Pipeline State
    setup_done: bool = Field(False, description="Flag indicating if the PyCaret setup() function has been completed.")
    setup_pipeline: Optional[Any] = Field(None, description="The PyCaret pipeline object after setup().")
    setup_config: Optional[Dict[str, Any]] = Field(None, description="A dictionary of parameters used for the setup() call.")

    # Model State
    model: Optional[Any] = Field(None, description="The last trained or tuned model object.")
    tuned_model: Optional[Any] = Field(None, description="A tuned version of the model.")
    final_model: Optional[Any] = Field(None, description="The final model object after finalize_model().")
    best_model: Optional[Any] = Field(None, description="The best model found from compare_models().")
    created_model: Optional[Any] = Field(None, description="The output of create_model().")
    ensemble_model: Optional[Any] = Field(None, description="The ensemble model created by ensemble_model().")

    # Leaderboards and Metrics
    leaderboard: Optional[Any] = Field(None, description="The dataframe output from compare_models().")
    tuned_leaderboard: Optional[Any] = Field(None, description="The dataframe output from tune_model().")
    
    # File Management
    model_path: Optional[str] = Field(None, description="File path to the saved model artifact.")
    model_name: Optional[str] = Field(None, description="Filename or custom name for the saved model.")
    
    # Conversation and UI State
    input_message: Optional[str] = Field(None, description="The latest message from the user.")
    last_output: Optional[str] = Field(None, description="The last message sent back to the user.")
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="A list of LangChain messages (HumanMessage, AIMessage, ToolMessage) for managing conversation memory."
    )
    
    # --- User Validation Flow ---
    user_validation_response: Optional[str] = Field(None, description="Stores the user's 'yes', 'no', or column name response during validation.")


    class Config:
        """
        Pydantic configuration settings.
        This allows us to store arbitrary objects (like PyCaret pipelines and models)
        in the model, which is necessary for our use case.
        """
        arbitrary_types_allowed = True

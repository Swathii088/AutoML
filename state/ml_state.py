ml_state = {
    "data": None,              # Loaded dataset
    "task": None,              # 'classification' or 'regression'
    "target": None,            # Target column

    "setup_done": False,       # True if setup() completed
    "setup_pipeline": None,    # Output of setup()
    "setup_config": None,      # Dict of used setup parameters

    "model": None,             # Last trained model (from create_model or tune_model)
    "tuned_model": None,       # Tuned version of the model
    "final_model": None,       # Output of finalize_model()

    "best_model": None,        # Output of compare_models() if used
    "leaderboard": None,       # Output of compare_models()

    "model_path": None,        # Path to saved model (for load/deploy)
    "model_name": None,        # Filename or custom name

    "created_model": None,     # Output of create_model()
    "tuned_leaderboard": None, # If tune_model was used
}

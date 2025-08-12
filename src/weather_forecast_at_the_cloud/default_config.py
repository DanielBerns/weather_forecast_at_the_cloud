import yaml

def generate_default_config() -> str:
    """
    Generates a default config.yaml string for the weather forecasting project.

    """
    # --- Define the configuration structure as a Python dictionary ---
    # This dictionary holds all the reconfigurable parameters for the training pipeline.
    config_data = {
        # Parameters for the training process
        'model_name': 'LSTM',  # Can be 'Linear', 'Dense', 'CNN', or 'LSTM'
        'epochs_per_session': 10,
        'sessions': 3, # Number of training sessions
        'patience': 3, # Patience for early stopping

        # Path configurations
        'data_path': './data',
        'models_path': './saved_models',
        'gdrive_path': '/content/drive/MyDrive/weather_forecast', # For Google Colab

        # Parameters for the data windowing and model architecture
        'out_steps': 24,
        'input_width': 24,
        'conv_width': 3, # Only used for the CNN model
        'label_column': 'temperature_celsius'
    }

    try:
        # yaml.dump() serializes the Python dictionary into YAML format.
        # default_flow_style=False makes it more readable (block style).
        # sort_keys=False preserves the order from the dictionary.
        yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

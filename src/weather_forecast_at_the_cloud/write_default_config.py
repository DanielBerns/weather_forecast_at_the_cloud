import yaml
from pathlib import Path

def generate_default_config() -> str:
    """
    Generates a default config.yaml string for the weather forecasting project.
    This centralized configuration allows for easy adjustments of hyperparameters,
    file paths, and model choices without altering the core logic of the
    application. Using a configuration file is a best practice for creating
    flexible and maintainable machine learning pipelines.
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
        'processed_data_path': './processed_data',
        'models_path': './models',
        'gdrive_path': '/content/drive/MyDrive/weather_forecast', # For Google Colab

        # Parameters for the data windowing and model architecture
        'out_steps': 24,
        'input_width': 24,
        'conv_width': 3, # Only used for the CNN model
        'label_column': 'temperature_celsius'
    }

    # yaml.dump() serializes the Python dictionary into YAML format.
    # default_flow_style=False makes it more readable (block style).
    # sort_keys=False preserves the order from the dictionary.
    return yaml.dump(config_data, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    config_yaml = Path("./", "config").with_suffix(".yaml").absolute()
    with open(config_yaml, "w") as text:
        content = generate_default_config()
        text.write(content)

import logging
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import data_handler
from utils.window_generator import WindowGenerator
from models.linear_weather_forecast import LinearWeatherForecast
from models.dense_weather_forecast import DenseWeatherForecast
from models.cnn_weather_forecast import CNNWeatherForecast
from models.rnn_weather_forecast import RNNWeatherForecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def plot_metrics(history, session):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label=f'Session {session} Train Loss')
    plt.plot(history.history['val_loss'], label=f'Session {session} Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def next_step():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- Load and Prepare Data ---
    logger.info("Step 1: Loading and cleaning data...")
    file_list = list(Path(config['data_path']).glob("*.csv"))
    raw_df = data_handler.load_data(file_list)
    cleaned_df = data_handler.clean_data(raw_df.copy())
    station_df = cleaned_df[cleaned_df['station_id'] == 'station_1'].copy()
    station_df.drop('station_id', axis=1, inplace=True)

    # --- Split the Data ---
    logger.info("\nStep 2: Splitting data...")
    n = len(station_df)
    train_df = station_df[0:int(n * 0.7)]
    val_df = station_df[int(n * 0.7):int(n * 0.9)]
    test_df = station_df[int(n * 0.9):]
    num_features = station_df.shape[1]

    # --- Normalize the Data ---
    logger.info("\nStep 3: Normalizing features...")
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / test_std

    # --- Create Data Windows ---
    logger.info("\nStep 4: Creating data windows...")
    multi_output_window = WindowGenerator(
        input_width=config['input_width'],
        label_width=config['out_steps'],
        shift=config['out_steps'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=[config['label_column']]
    )

    # --- Train Model ---
    models = {
        "Linear": LinearWeatherForecast,
        "Dense": DenseWeatherForecast,
        "CNN": CNNWeatherForecast,
        "LSTM": RNNWeatherForecast,
    }

    model_class = models.get(config['model_name'])
    if model_class is None:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    model_path = Path(config['models_path']) / config['model_name']
    if model_path.exists():
        logger.info(f"Loading existing model: {config['model_name']}")
        model = model_class.load(config['models_path'])
    else:
        logger.info(f"Creating new model: {config['model_name']}")
        if config['model_name'] == 'CNN':
            model = model_class(out_steps=config['out_steps'], num_features=num_features, conv_width=config['conv_width'])
        else:
            model = model_class(out_steps=config['out_steps'], num_features=num_features)

    for session in range(1, config.get('sessions', 3) + 1):
        logger.info(f"\n--- Training Session {session} ---")
        history = model.train(
            multi_output_window,
            epochs=config['epochs_per_session'],
            patience=config['patience'],
            verbose=1
        )
        model.save(config['models_path'])
        plot_metrics(history, session)

    logger.info("\n--- Final Evaluation ---")
    performance = model.model.evaluate(multi_output_window.test, verbose=0)
    logger.info(f"Test Loss (MSE): {performance[0]}, MAE: {performance[1]}")

if __name__ == '__main__':
    next_step()

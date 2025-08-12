from pathlib import Path
# from typing import Type

import tensorflow as tf
import numpy as np

from utils.window_generator import WindowGenerator

class RNNWeatherForecast:
    """
    A Recurrent Neural Network (RNN) model using LSTM for weather forecasting.

    This model implements the WeatherForecast protocol. It uses an LSTM layer
    to process sequences, allowing it to learn from long-term dependencies
    in the time-series data.
    """
    name = "LSTM"

    def __init__(self, out_steps: int, num_features: int):
        """
        Initializes the RecurrentWeatherForecast.

        Args:
            out_steps (int): The number of output steps to predict.
            num_features (int): The number of features in the input data.
        """
        self.out_steps = out_steps
        self.num_features = num_features
        self.model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # The LSTM layer returns only the last output in the sequence.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(out_steps*num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([out_steps, num_features])
        ])

    def train(
        self,
        window_generator: WindowGenerator,
        epochs: int = 20,
        patience: int = 2,
        verbose: int = 1
    ) -> tf.keras.Model:
        """
        Compiles and trains the LSTM model.

        Args:
            window_generator: The WindowGenerator instance providing the data.
            epochs: The number of epochs for training.
            patience: The patience for the EarlyStopping callback.
            verbose: Verbosity mode for training.

        Returns:
            The trained Keras model history.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True
        )

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

        history = self.model.fit(
            window_generator.train,
            epochs=epochs,
            verbose=verbose,
            validation_data=window_generator.val,
            callbacks=[early_stopping]
        )
        return history

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Makes a prediction using the trained model.

        Args:
            input_data: A numpy array of input features.

        Returns:
            A numpy array of predictions.
        """
        return self.model.predict(input_data)

    def save(self, path: str = "saved_models") -> None:
        """
        Saves the model to the specified path.

        Args:
            path: The directory to save the model in.
        """
        model_path = Path(path).absolute() / self.name
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path.with_suffix(".keras"))
        print(f"Model '{self.name}' saved to {model_path}")

    @classmethod
    def load(cls, path: str = "saved_models") -> "RNNWeatherForecast":
        """
        Loads a model from the specified path.

        Args:
            path: The directory to load the model from.

        Returns:
            An instance of RecurrentWeatherForecast with the loaded Keras model.
        """
        model_name = "LSTM"
        model_path = Path(path) / model_name

        # Assume default values for initialization.
        instance = cls(out_steps=24, num_features=7)
        loaded_keras_model = tf.keras.models.load_model(model_path)
        instance.model = loaded_keras_model
        print(f"Model '{model_name}' loaded from {model_path}")
        return instance

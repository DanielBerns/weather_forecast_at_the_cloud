from pathlib import Path
# from typing import Type

import tensorflow as tf
import numpy as np

from weather_forecast_at_the_cloud.utils.window_generator import WindowGenerator

class LinearWeatherForecast:
    """
    A simple linear model for weather forecasting.

    This model implements the WeatherModel protocol. It uses a single dense layer
    to predict the output based on the last time step of the input window.
    """
    name = "Linear"

    def __init__(self, out_steps: int, num_features: int):
        """
        Initializes the LinearWeatherForecast.

        Args:
            out_steps (int): The number of output steps to predict.
            num_features (int): The number of features in the input data.
        """
        self.out_steps = out_steps
        self.num_features = num_features
        self.model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(out_steps * num_features,
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
        Compiles and trains the Linear model.

        Args:
            window_generator: The WindowGenerator instance providing the data.
            epochs: The number of epochs for training.
            patience: The patience for the EarlyStopping callback.
            verbose: Verbosity mode for training.

        Returns:
            The trained Keras model.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
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
    def load(cls, path: str = "saved_models") -> "LinearWeatherForecast":
        """
        Loads a model from the specified path.

        Note: This is a simplified load method. For a complete restoration,
        you would also need to save/load `label_index` and `out_steps`.
        For this project, we'll re-initialize these when loading.

        Args:
            path: The directory to load the model from.

        Returns:
            An instance of LinearWeatherForecast with the loaded Keras model.
        """
        model_name = "Linear" # Hardcoded for this class
        model_path = Path(path) / model_name

        # A bit of a workaround for instantiation since we need parameters
        # that aren't saved with the model. In a full app, these would
        # be stored in a config file alongside the model.
        # We assume default values for now.
        instance = cls(out_steps=24, num_features=7)
        loaded_keras_model = tf.keras.models.load_model(model_path)
        instance.model = loaded_keras_model
        print(f"Model '{model_name}' loaded from {model_path}")
        return instance

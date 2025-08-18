from pathlib import Path
# from typing import Type

import tensorflow as tf
import numpy as np

from weather_forecast_at_the_cloud.utils.window_generator import WindowGenerator


class CNNWeatherForecast:
    """
    A Convolutional Neural Network (CNN) model for weather forecasting.

    This model implements the WeatherModel protocol. It uses a 1D convolutional
    layer to process sequences of input data, allowing it to learn from patterns
    over a specific time window.
    """
    name = "CNN"

    def __init__(self, out_steps: int, num_features: int, conv_width: int):
        """
        Initializes the ConvModel.

        Args:
            out_steps (int): The number of output steps to predict.
            num_features (int): The number of features in the input data.
            conv_width (int): The width of the convolution window.
        """
        self.out_steps = out_steps
        self.num_features = num_features
        self.conv_width = conv_width
        self.model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            # Shape => [batch, 1, 256]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(conv_width)),
            # Shape => [batch, 1, out_steps*features]
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
        Compiles and trains the CNN model.

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
    def load(cls, path: str = "saved_models") -> "CNNWeatherForecast":
        """
        Loads a model from the specified path.

        Args:
            path: The directory to load the model from.

        Returns:
            An instance of ConvModel with the loaded Keras model.
        """
        model_name = "CNN"
        model_path = Path(path) / model_name

        # Assume default values for initialization.
        # A real application would use a config file.
        instance = cls(out_steps=24, num_features=7, conv_width=3)
        loaded_keras_model = tf.keras.models.load_model(model_path)
        instance.model = loaded_keras_model
        print(f"Model '{model_name}' loaded from {model_path}")
        return instance

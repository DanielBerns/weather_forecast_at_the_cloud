from pathlib import Path

import tensorflow as tf
import numpy as np

from weather_forecast_at_the_cloud.utils.window_generator import WindowGenerator

class DenseWeatherForecast:
    """
    A multi-layer dense neural network for weather forecasting.

    This model implements the WeatherForecast protocol. It uses two dense hidden
    layers with ReLU activation to learn non-linear relationships in the data.
    """
    name = "Dense"

    def __init__(self, out_steps: int, num_features: int):
        """
        Initializes the DenseWeatherForecast.

        Args:
            out_steps (int): The number of output steps to predict.
            num_features (int): The number of features in the input data.
        """
        self.out_steps = out_steps
        self.num_features = num_features
        self.model = tf.keras.models.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, 64]
            tf.keras.layers.Dense(64, activation='relu'),
            # Shape => [batch, 1, 64]
            tf.keras.layers.Dense(64, activation='relu'),
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
        Compiles and trains the Dense model.

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
    def load(cls, path: str = "saved_models") -> "DenseWeatherForecast":
        """
        Loads a model from the specified path.

        Args:
            path: The directory to load the model from.

        Returns:
            An instance of DenseWeatherForecast with the loaded Keras model.
        """
        model_name = "Dense"
        model_path = Path(path) / model_name

        # As before, we assume default values for initialization
        # since they are not saved with the Keras model.
        # A real application would use a config file.
        instance = cls(out_steps=24, num_features=7)
        loaded_keras_model = tf.keras.models.load_model(model_path)
        instance.model = loaded_keras_model
        print(f"Model '{model_name}' loaded from {model_path}")
        return instance

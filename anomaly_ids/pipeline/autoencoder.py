import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AutoencoderIDS:
    """
    Autoencoder used for anomaly detection.
    Trained ONLY on normal traffic.
    Provides:
    - encoded representations
    - reconstruction error scores (This is not getting used currently in the pipeline ahead)
    """

    def __init__(self, input_dim: int, encoding_dim: int = 32, dropout: float = 0.2, epochs: int = 20, batch_size: int = 256): # 'Input Dimensions' will always have to be passed
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

        self.autoencoder = None # Combining encoding, decoding
        self.encoder = None # Only decoding
        self.history = None
        self.weights_loaded = False  # Track if trained weights were loaded

        self._build_model()

    # Model Architecture
    def _build_model(self):
        encoder_input = keras.Input(shape = (self.input_dim,))

        # Encoder
        # Suppressing the features till 32 (Model will start learning from here only, i.e at the end of encoder)
        x = layers.Dense(128, activation = "relu")(encoder_input) # Upscales 107 features to 128 features for neural net processing ahead
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation = "relu")(x)
        x = layers.Dropout(self.dropout)(x)
        encoded = layers.Dense(self.encoding_dim, activation = "relu")(x)

        # Decoder
        # Model will start to learn from 32 features, and will build all the way uptil 121/122 features, thereby again upscaling till 128 features
        x = layers.Dense(64, activation = "relu")(encoded)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(128, activation = "relu")(x)
        decoded = layers.Dense(self.input_dim, activation = "linear")(x)

        # Full AutoEncoder
        self.autoencoder = keras.Model(encoder_input, decoded)
        self.autoencoder.compile(optimizer = "adam", loss = "mse")

        # Encoder-only Model
        self.encoder = keras.Model(encoder_input, encoded)

    # Training the model(Fit only Normal Data)
    def fit(self, X_normal_train, X_normal_val = None, verbose = 0):
        """
            Train AutoEncoder on Normal data only (Make it learn what is normal)
        """
        if X_normal_val is None:
            validation_data = None
        else:
            validation_data = (X_normal_val, X_normal_val) # start validation on normal data and evaluate on normal data also

        self.history = self.autoencoder.fit(
            X_normal_train,
            X_normal_train,
            epochs = self.epochs,
            batch_size = self.batch_size,
            validation_data = validation_data,
            verbose = verbose
        )

        self.weights_loaded = True  # Mark that we have trained weights

        return self

    # Encoded Features (These will be used by the supervised model)
    def encode(self, X):
        return self.encoder.predict(X, verbose = 0)

    # Reconstruction Error (Error made by the AutoEncoder in the process of encoding and decoding)
    # MSE (Mean Squared Error) will be used for evaluation
    # This will be high for anomalous data
    def reconstruction_error(self, X):
        recon = self.autoencoder.predict(X, verbose = 0)
        return np.mean(np.square(X - recon), axis = 1)

    # Convenience Method (Track loss made by the autoencoder)
    def get_last_train_loss(self):
        if self.history is None:
            return None
        return self.history.history["loss"][-1]

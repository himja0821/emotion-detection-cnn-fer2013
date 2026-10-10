"""CNN architecture for FER-2013 emotion classification.

A compact VGG-style network: stacked ``Conv2D -> BatchNorm -> ReLU`` blocks
followed by max-pooling, then dense layers with dropout for classification.
Light on-the-fly augmentation layers are included so the model is robust to
the small pose/lighting variations in FER-2013.
"""

from tensorflow import keras
from tensorflow.keras import layers

from . import config


def _conv_block(x, filters: int, dropout: float = 0.0):
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def build_cnn(input_shape=None, num_classes: int = None, augmentation: bool = True):
    """Create and compile the emotion CNN.

    Parameters
    ----------
    input_shape : tuple
        ``(height, width, channels)``. Defaults to ``(48, 48, 1)``.
    num_classes : int
        Number of emotion classes (7 for FER-2013).
    augmentation : bool
        If True, lightweight augmentation layers are added *inside* the model.
        They are active only during training.
    """
    input_shape = input_shape or (config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS)
    num_classes = num_classes or config.NUM_CLASSES

    inputs = keras.Input(shape=input_shape, name="face")
    x = inputs

    if augmentation:
        # Light geometric jitter only. Horizontal flip is intentionally skipped:
        # facial expressions are asymmetric (e.g. disgust), so mirroring hurts.
        x = layers.RandomRotation(0.05, name="aug_rot")(x)
        x = layers.RandomTranslation(0.05, 0.05, name="aug_shift")(x)
        x = layers.RandomZoom(0.05, name="aug_zoom")(x)

    # Block 1
    x = _conv_block(x, 32)
    x = _conv_block(x, 32)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # Block 2
    x = _conv_block(x, 64)
    x = _conv_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # Block 3
    x = _conv_block(x, 128)
    x = _conv_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    # Classifier head
    x = layers.Flatten()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="emotion_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(path: str = None):
    """Load a saved Keras model (``.keras`` or ``.h5``)."""
    import os

    path = path or config.MODEL_PATH
    if not os.path.exists(path):
        alt = config.MODEL_H5_PATH
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"No trained model at {path}. Train one first: python -m src.train"
            )
    return keras.models.load_model(path)


if __name__ == "__main__":
    m = build_cnn()
    m.summary()

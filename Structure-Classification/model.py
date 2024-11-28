import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator

# Keras EfficientNet import (if needed, but be cautious with compatibility)
from keras_efficientnets import EfficientNetB0

def create_model():
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(6, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()
model.compile(optimizer=Adam(learning_rate=0.00001),loss='categorical_crossentropy', metrics=['accuracy'])

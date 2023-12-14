import tensorflow as tf
import keras
import os
import pathlib

name="small_v6_noShuffle-05"
modelFiles = "models"
litModelFiles = "liteModels"


model = keras.models.load_model(os.path.abspath(f"{modelFiles}/{name}.keras"))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path(os.path.abspath(f"{litModelFiles}"))
tflite_model_file = tflite_models_dir/f"{name}.tflite"
tflite_model_file.write_bytes(tflite_model)


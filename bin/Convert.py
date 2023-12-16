import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import keras

name="small_v7_noVal"
modelFiles = "concreteModels"

targetDir = "liteModels"
targetName = "small_v7_noVal_lite"
run = "lite"

if run =="concrete":
    model = keras.models.load_model(os.path.abspath(f"{modelFiles}/{name}.keras"))
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 42), dtype=tf.int64)])  # Adjust the input shape and dtype as per your model's requirements
    def tf_func_call(self, inp):
        return model(inp)

    concrete_function = tf_func_call.get_concrete_function()


    constant_graph = convert_variables_to_constants_v2(concrete_function)

    module = tf.Module()
    module.func = constant_graph
    tf.saved_model.save(module, os.path.abspath(f"concreteModels/{name}"))
    # loaded_module = tf.saved_model.load("model_internal")
    # concrete_function = loaded_module.func

    # constant_graph(tf.constant([[3, 1.2]]))

if run=="lite":
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.abspath(f"{modelFiles}/{name}"))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    # Save the model.
    with open(os.path.abspath(f"{targetDir}/{targetName}.tflite"), 'wb') as f:
        f.write(tflite_quant_model)
    

import tensorflow as tf
import os

name="small_v6_noShuffle-05"
litModelFiles = "liteModels"


modelPath = os.path.abspath(f"{litModelFiles}/{name}.tflite")

interpreter = tf.lite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, )

# Run inference.
interpreter.invoke()

# Post-processing: remove batch dimension and find the digit with highest
# probability.
output = interpreter.tensor(output_index)
print(output)
digit = np.argmax(output()[0])


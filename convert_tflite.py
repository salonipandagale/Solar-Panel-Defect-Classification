import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Loading SavedModel...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    "solar_model_deploy"
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting to TFLite...")

tflite_model = converter.convert()

with open("solar_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully.")

print(
    "Size:",
    round(len(tflite_model) / (1024 * 1024), 2),
    "MB"
)

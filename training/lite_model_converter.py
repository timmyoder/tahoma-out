import tensorflow as tf

from config import MODELS_DIR

model_name = 'simple_cnn_weighted'
# Convert the model
model_dir = MODELS_DIR / 'to_convert'
converter = tf.lite.TFLiteConverter.from_saved_model(str(model_dir))
tflite_model = converter.convert()

# Save the model.
with open(MODELS_DIR / 'lite' / f'{model_name}.tflite', 'wb') as out_file:
    out_file.write(tflite_model)

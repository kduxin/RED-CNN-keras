import keras
import sys

name = sys.argv[1]
model = keras.models.load_model(name)
suffix = name[:name.rfind('.')]
model.save_weights(suffix + '.weights')
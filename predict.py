"""
predict.py
==========
Quick single-image prediction using the trained model.

Run:
    python predict.py path/to/image.jpg
"""

import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH = "cat_dog_improved.keras"
IMG_SIZE = (64, 64)

if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit(1)

img_path = sys.argv[1]

model = load_model(MODEL_PATH)

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

score = float(model.predict(img_array, verbose=0)[0][0])

# flow_from_directory maps folders alphabetically: cats=0, dogs=1
label = "dog" if score > 0.5 else "cat"
confidence = score if score > 0.5 else (1 - score)

print(f"Prediction: {label}  (sigmoid={score:.4f}, confidence={confidence*100:.1f}%)")

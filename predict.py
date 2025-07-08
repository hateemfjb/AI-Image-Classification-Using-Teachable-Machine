import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

model = tensorflow.keras.models.load_model('keras_model.h5')

with open("labels.txt", "r") as f:
    class_names = f.read().splitlines()

image = Image.open("test.jpg").convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.expand_dims(normalized_image_array, axis=0)

prediction = model.predict(data)
predicted_class = class_names[np.argmax(prediction)]
print("Prediction:", predicted_class)
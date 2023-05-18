import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# AFIȘAREA VERSIUNILOR LIBRĂRIILOR
print("Versiune OpenCV: ", cv2.__version__)
print("Versiune TensorFlow: ", tf.__version__)

# Calea către fișierul .h5 al modelului antrenat
model_path = 'model.h5'

# Dimensiunile imaginilor de test
width = 80
height = 80

# Directorul cu imaginile de test
test_dir = '../dataset/test'

# Numărul de clase
num_classes = 2

# Funcția de decodare și preprocesare a imaginilor
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (width, height))
    img = img / 255.0  # Normalizare la interalul [0, 1]
    return img

# Încărcarea modelului antrenat
model = keras.models.load_model(model_path)

# Crearea generatorului de date pentru testare
test_datagen = ImageDataGenerator(rescale=1./255)

# Încărcarea imaginilor de test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(width, height),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Realizarea predicțiilor pe setul de test
predictions = model.predict(test_generator)

# Conversia predicțiilor în clase
predicted_classes = np.argmax(predictions, axis=1)

# Afișarea predicțiilor
class_names = {v: k for k, v in test_generator.class_indices.items()}

for i in range(len(predicted_classes)):
    image_path = os.path.join(test_dir, test_generator.filenames[i])
    predicted_class = class_names[predicted_classes[i]]
    print("Imagine:", image_path, "- Clasa prezisă:", predicted_class)

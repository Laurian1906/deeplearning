"""
Proiect: Extragerea datelor biometrice din amprentele digitale
Modul: Analiza conținutului imaginilor
Student: Hurduza Laurian

main.py: Antrenare si validare.
"""

# INCLUDEREA LIBRĂRIILOR NECESARE
import cv2
import tensorflow
import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from skimage.feature import hog

# AFIȘAREA VERSIUNILOR LIBRĂRIILOR
print("Versiune OpenCV: ", cv2.__version__);
print("Versiune Tensorflow: ", tensorflow.__version__);

# EXTRAGEREA IMAGINILOR DIN DIRECTORUL dataset

# variabile
width = 80
height = 80

# se specifica locatia imaginilor pentru antrenament
input_dir = 'dataset/train/original_train'

# in acest director se vor salva imaginile redimensionate
output_dir = 'dataset/resized_train'

# in acest director se vor salva imaginile pe care s-a aplica functia hog()
hog_dir = 'dataset/train/hog_train'

# daca nu exista fisierul se va crea
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""
Etapa de preprocesare a imaginilor
"""

# REDIMENSIONAREA IMAGINILOR

# se parcurge fiecare imagine din directorul train
for filename in os.listdir(input_dir):
    # citirea imaginii
    img = cv2.imread(os.path.join(input_dir, filename))

    # redimensioneaza imaginea
    resized_img = cv2.resize(img, (width, height))

    # salvarea imaginii in directorul specificat
    cv2.imwrite(os.path.join(output_dir, filename), resized_img)

if os.path.exists(output_dir):
    print("S-au citit imaginile din directorul: ", input_dir)
    print("S-au redimensionat imaginile din directorul: ", input_dir)
    print("Imaginile redimensionate au fost salvate in fisierul: ", output_dir)

# se creeaza directorul hog_train daca nu exista
if not os.path.exists(hog_dir):
    os.makedirs(hog_dir)

# se parcurge fiecare imagine din directorul resized_train
for filename in os.listdir(output_dir):
    # citirea imaginii
    img = cv2.imread(os.path.join(output_dir, filename))

    # se convertesc imaginile in alb-negru
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # se aplica hog pe fiecare imagine
    hog_img = hog(img, channel_axis=-1, orientations=9)

    # se salveaza imaginea in directorul hog_train
    cv2.imwrite(os.path.join(hog_dir, filename), hog_img)

if os.path.exists(hog_dir):
    print("S-a aplicat functia hog pe imaginile din fisierul: ", output_dir)
    print("Imaginile cu hog au fost salvate in fisierul: ", hog_dir)

# NORMALIZAREA DATELOR

# Etape preliminatorii

img_size = 80
np.random.seed(42)
labels = []
image_paths = []

for filename in os.listdir(hog_dir):
    img_path = os.path.join(hog_dir, filename)
    image_paths.append(img_path)
    label = filename.split("_")[0]  # numele clasei din numele fisierului
    labels.append(label)

labels = np.array(labels)
one_hot_labels = tf.one_hot(labels, depth=2)
num_images = len(image_paths)

# Funcția care va fi folosită pentru a decoda imaginile din fișier
def decode_img(img_path):
    # Citim imaginea folosind TensorFlow
    img = tf.io.read_file(img_path)
    # Decodăm imaginea folosind TensorFlow
    img = tf.image.decode_jpeg(img, channels=1)
    # Convertim imaginea la o dimensiune specifică
    img = tf.image.resize(img, [img_size, img_size])
    # Convertim imaginea la tipul float32
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Normalizăm imaginea
    img = (img - 0.5) / 0.5
    return img


def create_dataset(image_paths, labels):
    # Convertim etichetele în codificare one-hot
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=2)

    # Folosim lista image_paths și vectorul one_hot_labels pentru a crea un obiect de tip dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, one_hot_labels))

    # Funcție pentru citirea și preprocesarea imaginilor
    def preprocess_image(filename, label):
        # Citirea imaginii
        img = tf.io.read_file(filename)
        # Decodarea imaginii folosind formatul jpeg
        img = tf.image.decode_jpeg(img, channels=3)
        # Redimensionarea imaginii la dimensiunea dorită
        img = tf.image.resize(img, (width, height))
        # Convertirea valorilor pixelilor la intervalul [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        # Returnarea perechii (imagine, label)
        return img, label

    # Aplicăm funcția de preprocesare pe fiecare element din dataset
    dataset = dataset.map(preprocess_image)

    return dataset


# Generăm vectorii pentru train, valid și test folosind funcția `train_test_split` din scikit-learn
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

train_end = int(train_ratio * num_images)
valid_end = int((train_ratio + valid_ratio) * num_images)

train_image_paths = image_paths[:train_end]
train_labels = one_hot_labels[:train_end]

valid_image_paths = image_paths[train_end:valid_end]
valid_labels = one_hot_labels[train_end:valid_end]

test_image_paths = image_paths[valid_end:]
test_labels = one_hot_labels[valid_end:]

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_dir = 'dataset/train'
real_dir = 'dataset/validation'

batch_size = 10

train_dataset = datagen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',shuffle=True)

valid_dataset = datagen.flow_from_directory(
    real_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',shuffle=True)
"""
Construierea si antrenarea modelului
"""
class_names=train_dataset.class_indices
print(class_names)
num_classes = len(class_names)
epochs = 10
# Construirea modelului
model = keras.Sequential([
    layers.Rescaling(1./255,input_shape=(height,width,3)),
    layers.Conv2D(8,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(num_classes),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1),
    layers.Flatten(),
    layers.Dropout(0.2),  # adauga dropout dupa stratul Flatten()
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')
])

# Compilarea modelului
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Antrenarea modelului
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs
)

# Evaluarea performantei modelului
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Realizarea predictiilor
# predictions = model.predict()

"""
Extragerea datelor biometrice
"""

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
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

# INCLUDEREA VARIABILELOR DIN ALTE SCRIPTURI
from classifier import predicted_classes

# AFIȘAREA VERSIUNILOR LIBRĂRIILOR
print("Versiune OpenCV: ", cv2.__version__);
print("Versiune Tensorflow: ", tensorflow.__version__);

# EXTRAGEREA IMAGINILOR DIN DIRECTORUL dataset

# variabile
width = 80
height = 80

# se specifica locatia imaginilor pentru antrenament
input_dir = '../dataset/train/original_train'

# in acest director se vor salva imaginile redimensionate
output_dir = '../dataset/resized_train'

# in acest director se vor salva imaginile pe care s-a aplica functia hog()
hog_dir = '../dataset/train/hog_train'

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
features = []
for filename in os.listdir(output_dir):
    # citirea imaginii
    img = cv2.imread(os.path.join(output_dir, filename))

    # se convertesc imaginile in alb-negru
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # se aplica hog pe fiecare imagine
    features, hog_img = hog(img, channel_axis=-1, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)

    # se salveaza imaginea in directorul hog_train
    cv2.imwrite(os.path.join(hog_dir, filename), hog_img)

np.save('features.npy', features)

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
np.save('labels.npy', labels)
# Verificăm tipul etichetelor
if np.issubdtype(labels.dtype, np.number):
    numeric_labels = labels
else:
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

num_classes = len(np.unique(numeric_labels))
one_hot_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes=num_classes)
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
    # Folosim lista image_paths și vectorul one_hot_labels pentru a crea un obiect de tip dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Funcție pentru citirea și preprocesarea imaginilor
    def preprocess_image(filename, label):
        # Citirea imaginii
        img = decode_img(filename)
        # Returnarea perechii (imagine, label)
        return img, label

    # Aplicăm funcția de preprocesare pe fiecare element din dataset
    dataset = dataset.map(preprocess_image)

    return dataset


# Generăm vectorii pentru train, valid și test
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

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_dir = '../dataset/train'
real_dir = '../dataset/validation'

batch_size = 20

train_dataset = datagen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training', shuffle=True)

valid_dataset = datagen.flow_from_directory(
    real_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation', shuffle=True)

"""
Construierea si antrenarea modelului
"""
# class_names = train_dataset.class_indices
class_names = predicted_classes
print(class_names)
num_classes = len(class_names)
epochs = 20
# Construirea modelului
model = keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=(height, width, 3)),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='sigmoid'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='sigmoid'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='tanh'),
    layers.MaxPooling2D(),
    layers.Dense(128, activation='relu'),

    layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(num_classes, activation='relu')
])

# Compilarea modelului
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

model.save('model.h5')

# Realizarea predictiilor
# predictions = model.predict()

"""
Extragerea datelor biometrice
"""


# Importam librariile neceasre
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras

# caracteristici HOG și etichete
caracteristici_hog = np.load('features.npy')
etichete = np.load('labels.npy')

label_encoder = LabelEncoder()
etichete_encode = label_encoder.fit_transform(etichete)

X_train, X_test, y_train, y_test = train_test_split(caracteristici_hog, etichete_encode, test_size=0.2, random_state=42)

num_epochs = 10
batch_size = 20
lungime_caracteristici_hog = len(caracteristici_hog)
num_classes = len(etichete)

model = keras.models.load_model('model.h5')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])




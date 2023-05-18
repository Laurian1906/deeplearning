import keras
from keras.applications.densenet import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.python.keras import regularizers

# Definirea arhitecturii modelului
model = Sequential()

# Adăugarea stratului de convoluție
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))

# Adăugarea stratului de max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adăugarea altor straturi de convoluție și max pooling, după necesitate
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Transformarea matricei 2D într-un vector 1D
model.add(Flatten())

# Adăugarea straturilor fully connected
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilarea modelului
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Sumarul arhitecturii modelului
model.summary()

# Setarea cailor catre seturile de date
train_dir = 'preprocessed_dataset/train'
validation_dir = 'preprocessed_dataset/validation'
test_dir = 'preprocessed_dataset/test'

# Setarea numarului de imagini in fiecare set
num_train_images = 1155
num_validation_images = 734
num_test_images = 176

# Setarea parametrilor modelului
input_shape = (80, 80, 1)
num_classes = 1

# Setarea hiperparametrilor de antrenare
batch_size = 32
epochs = 10

# Construirea modelului CNN
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(num_classes, activation='sigmoid'))

# Construirea modelului
model = keras.Sequential([
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Crearea generatorilor de date pentru seturile de antrenare, validare si testare
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(80, 80),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(80, 80),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(80, 80),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

# Antrenarea modelului
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_images // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_validation_images // batch_size
)

# Evaluarea modelului pe setul de testare
test_loss, test_accuracy = model.evaluate(test_generator, steps=num_test_images // batch_size)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

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


print('TESTARE:')
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)


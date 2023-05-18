import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Directorul în care sunt stocate imaginile HOG
hog_dir = '../dataset/train/hog_train'

# Dimensiunea dorită a caracteristicilor HOG
desired_feature_size = 8100

# Citirea caracteristicilor HOG și etichetelor claselor
features = []
labels = []

# Se parcurg imaginile HOG
for filename in os.listdir(hog_dir):
    # Citirea imaginii HOG
    img = cv2.imread(os.path.join(hog_dir, filename), cv2.IMREAD_GRAYSCALE)

    # Extrage caracteristicile HOG
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # Verifică dimensiunea caracteristicilor HOG
    if hog_features.shape[0] == desired_feature_size:
        # Adăugarea caracteristicilor și etichetei în liste
        features.append(hog_features)
        labels.append(filename.split("_")[0])  # Extrage clasa din numele fișierului

# Transformarea listelor în matrice numpy
X_train = np.vstack(features)
y_train = np.array(labels)

# Împărțirea setului de date în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Inițializarea unui clasificator SVM
classifier = svm.SVC()

# Antrenarea clasificatorului SVM
classifier.fit(X_train, y_train)

# Realizarea predicțiilor pe setul de testare
y_pred = classifier.predict(X_test)

# Obținerea claselor prezise
predicted_classes = classifier.predict(X_test)

# Afișarea claselor prezise și claselor reale
for i in range(len(X_test)):
    print("Clasa prezisă:", predicted_classes[i])
    print("Clasa reală:", y_test[i])
    print("-----")

# Evaluarea performanței modelului
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


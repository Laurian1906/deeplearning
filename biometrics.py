import cv2
import numpy as np
from matplotlib import pyplot as plt


# Funcție pentru identificarea punctelor de interes din amprentă
def detect_keypoints(image):
    # Aplică algoritmul ORB pentru detectarea punctelor de interes
    orb = cv2.ORB_create()
    keypoints = orb.detect(image, None)
    return keypoints


# Funcție pentru calcularea direcției liniei ridge în fiecare punct
def compute_orientation(image, keypoints):
    # Calculează direcția liniei ridge în fiecare punct cheie
    orientations = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        # Extrage o fereastră vecină în jurul punctului cheie
        window_size = 16
        window = image[int(y) - window_size:int(y) + window_size, int(x) - window_size:int(x) + window_size]
        # Calculează gradientul imaginii în fereastra vecină
        gradient_x = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
        # Calculează direcția liniei ridge utilizând gradientul
        orientation = np.arctan2(gradient_y.mean(), gradient_x.mean())
        orientations.append(orientation)
    return orientations


# Imaginea prelucrată a amprentei
preprocessed_image = cv2.imread('filtered_image.jpg', 0)
if preprocessed_image is None:
    print("Nu s-a putut încărca imaginea!")


# Identifică punctele de interes
keypoints = detect_keypoints(preprocessed_image)

# Calculează direcția liniei ridge în fiecare punct
orientations = compute_orientation(preprocessed_image, keypoints)

# Afișează rezultatele
for i, keypoint in enumerate(keypoints):
    print("Keypoint {}: Coordonate={}, Orientare={}".format(i + 1, keypoint.pt, orientations[i]))

print("Număr de puncte cheie detectate:", len(keypoints))
print("Număr de orientări calculate:", len(orientations))
print("Coordonatele punctelor cheie:", [keypoint.pt for keypoint in keypoints])
print("Orientările calculate:", orientations)

print("Done")

# Afiseaza grafic numarul de puncte cheie detectate
plt.figure()
plt.title("Număr de puncte cheie detectate")
plt.bar(['Puncte cheie'], [len(keypoints)])
plt.show()

# Afiseaza grafic orientările calculate
plt.figure()
plt.title("Orientări calculate")
plt.plot(range(len(orientations)), orientations)
plt.xlabel("Index punct cheie")
plt.ylabel("Orientare")
plt.show()
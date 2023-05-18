import cv2
import numpy as np
from matplotlib import pyplot as plt

# Incarcare imagine
image = cv2.imread('dataset/train/original_train/00003_65.bmp', cv2.IMREAD_GRAYSCALE)

# PREPROCESARE

# Redimensionare imagine
resized_image = cv2.resize(image, (80, 80))

# Corectie de contrast
equalized_image = cv2.equalizeHist(resized_image)

# Eliminare zgomot
denoised_image = cv2.GaussianBlur(equalized_image, (1, 1), 0)

# Umplerea golurilor din imagine
kernel = np.ones((5,5), np.uint8) # filtrul prin care imaginea va fi trecuta
opened_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

# Subtierea liniei amprentei
skeletonized_image = cv2.morphologyEx(closed_image, cv2.MORPH_HITMISS, kernel)

# Detectarea marginilor
edges = cv2.Canny(skeletonized_image, threshold1=50, threshold2=150)

# Binarizarea imaginii
_, binary_image = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Normalizarea imaginii
normalized_image = binary_image.astype(np.float32) / 255.0

# Transformari geometrice
height, width = normalized_image.shape[:2] # obtinerea dimensiunilor imaginii
M = cv2.getRotationMatrix2D((width /2, height/2), 30, 1) # rotire la 30 grade
transformed_image = cv2.warpAffine(normalized_image, M, (width, height))

# Segmentarea bazata pe praguri multiple
threshold_lower = 12
threshold_upper = 5000
segmented_image = cv2.inRange(binary_image, threshold_lower, threshold_upper)

# Conturul
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # gasirea contururilor
contour_image = np.zeros_like(transformed_image) # crearea unei imagini pentru afisarea contururlor
cv2.drawContours(contour_image, contours, -1, (255,255,255), thickness=1) # desenarea conturului

# Filtrul Gabor ( detectarea liniilor amprentei )
#parametrii
ksize = 31  # Dimensiunea kernelului
sigma = 3  # Deviația standard a gaussianei
theta = 0  # Unghiul de orientare al filtrului
lambd = 10  # Lungimea de undă a sinusoidului
gamma = 0.5  # Aspectul (raportul dintre lungimea de undă și deviația standard)

#generare filtru gabor
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
#aplicare filtru gabor
filtered_image = cv2.filter2D(contour_image, cv2.CV_64F, kernel)

# Aplicare Watershed pentru segmentarea regiunilor de interes
dist_transform = cv2.distanceTransform(segmented_image, cv2.DIST_L2, 3)
_, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
filtered_image = filtered_image.astype(np.uint8)  # Conversie la tipul de date uint8
sure_fg = sure_fg.astype(np.uint8)  # Conversie la tipul de date uint8
unknown = cv2.subtract(filtered_image, sure_fg)


_, markers = cv2.connectedComponents(sure_fg)

markers += 1
markers[unknown == 255] = 0

transformed_image = transformed_image.astype(np.uint8)
markers = markers.astype(np.int32)
transformed_image_color = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2BGR)
filtered = cv2.watershed(transformed_image_color, markers)
filtered_image[filtered == -1] = 255

print(markers)
final = cv2.imwrite('filtered_image.jpg', filtered_image)

# Crearea unui plot si afisarea imaginii
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12,8))

axes[0,0].imshow(image)
axes[0,1].imshow(resized_image)
axes[0,2].imshow(equalized_image)
axes[0,3].imshow(denoised_image)
axes[1,0].imshow(opened_image)
axes[1,1].imshow(closed_image)
axes[1,2].imshow(skeletonized_image)
axes[1,3].imshow(binary_image)
axes[2,0].imshow(normalized_image)
axes[2,1].imshow(transformed_image)
axes[2,2].imshow(segmented_image)
axes[2,3].imshow(contour_image)
axes[3,0].imshow(filtered_image)
plt.tight_layout()
plt.axis('off')
plt.show()
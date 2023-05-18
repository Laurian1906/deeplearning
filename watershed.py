import os
import cv2
import numpy as np

# Directorul care conține imaginile de test preprocesate
test_directory = 'preprocessed_dataset/test/test_i'
test_output = 'watershed/'

# Iterează prin imaginile din director
for filename in os.listdir(test_directory):
    if filename.endswith('.bmp'):
        # Încarcă imaginea de test preprocesată
        image_path = os.path.join(test_directory, filename)
        preprocessed_test_image = cv2.imread(image_path, 0)

        # Aplică segmentarea și Watershed pe imaginea de test preprocesată
        dist_transform = cv2.distanceTransform(preprocessed_test_image, cv2.DIST_L2, 3)
        _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        preprocessed_test_image = preprocessed_test_image.astype(np.uint8)  # Conversie la tipul de date uint8
        sure_fg = sure_fg.astype(np.uint8)  # Conversie la tipul de date uint8
        unknown = cv2.subtract(preprocessed_test_image, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0

        preprocessed_test_image = preprocessed_test_image.astype(np.uint8)
        markers = markers.astype(np.int32)
        preprocessed_test_image_color = cv2.cvtColor(preprocessed_test_image, cv2.COLOR_GRAY2BGR)
        filtered = cv2.watershed(preprocessed_test_image_color, markers)
        preprocessed_test_image[filtered == -1] = 255

        output_path = os.path.join(test_output, filename)
        cv2.imwrite(output_path, preprocessed_test_image)

print("Watershed has been applied to all test images")
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
image = cv2.imread('/content/drive/MyDrive/chess.jfif', cv2.IMREAD_GRAYSCALE)
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
corners = cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.intp(corners)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image_color, (x, y), 5, (0, 255, 0), -1)
cv2_imshow(image_color)

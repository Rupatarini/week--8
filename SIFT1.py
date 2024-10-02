import cv2
import matplotlib.pyplot as plt
image = cv2.imread('/content/drive/MyDrive/nvgr.jpeg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
key points, descriptors = sift.detectAndCompute(image, None)
output_image = cv2.drawKeypoints(image, key points, None)
plt.figure(figsize=(10, 5))
plt.imshow(output_image, cmap='gray')
number=len(key points)
plt.title('SIFT Keypoints')
plt.axis('off')
plt.show()
print(f"The number of keypoints are {number}")

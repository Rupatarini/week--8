import cv2
import numpy as np
from matplotlib import pyplot as plt
def display_image(image, title='Image'):
    plt.figure(figsize=(15, 10))  # Increased figure size for better visibility
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
# Load the first image
image_path1 = '/content/drive/MyDrive/nvgr.jpeg'  # Ensure this path is correct
image1 = cv2.imread(image_path1)
# Load the second image
image_path2 = '/content/drive/MyDrive/mvgr.png'  # Corrected image path
image2 = cv2.imread(image_path2)
# Check if both images were loaded successfully
if image1 is None:
    print(f"Error: Could not open or find the image at {image_path1}.")
    exit()
if image2 is None:
    print(f"Error: Could not open or find the image at {image_path2}.")
    exit()
# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# Initialize SIFT detector
sift = cv2.SIFT_create()
# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)
# Check if descriptors are found
if descriptors1 is None:
    print("Error: No descriptors found in the first image.")
    exit()
if descriptors2 is None:
    print("Error: No descriptors found in the second image.")
    exit()
# Initialize BFMatcher with default parameters
bf = cv2.BFMatcher()
# Perform KNN matching with k=2
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
# Apply ratio test as per Lowe's paper
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
# Draw only the good matches
matched_image = cv2.drawMatches(
    image1, keypoints1,
    image2, keypoints2,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Display the matched image
display_image(matched_image, 'Matched Image')
# Print the number of good matches
print(f"The number of good matches are {len(good_matches)}")

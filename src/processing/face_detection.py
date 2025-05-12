import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import cv2

# 1. Load images from a folder
def load_images(folder_path, image_size=(10, 10)):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pgm') or filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, image_size)
            images.append(img_resized.flatten())
    return np.array(images)

# Set your dataset folder path here
dataset_path = 'D:/cv_task5_test/s1'
image_vectors = load_images(dataset_path)   # returns N_images x D_pixels np array , where D is the num of pixels which is 90000 here.
print(f"Loaded {image_vectors.shape[0]} images, each with {image_vectors.shape[1]} pixels (dimensions).")

# 2. Mean center the data
mean_face = np.mean(image_vectors, axis=0)
centered_data = image_vectors - mean_face

# 3. Compute PCA (eigenfaces)
n_components = 10  # number of kept components (eigan faces)
pca = PCA(n_components=n_components, whiten=True)
pca.fit(centered_data)

eigenfaces = pca.components_
print(eigenfaces.shape)

# 4. Visualize the eigenfaces
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i].reshape((10, 10)), cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')
plt.tight_layout()
plt.show()


test_img = cv2.imread('img_5.jpg', cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (10, 10))
test_img = test_img.flatten()
centered_test_img = (test_img - mean_face).reshape(1, -1)

projected_test_img = pca.transform(centered_test_img)
reconstructed_test_img = pca.inverse_transform(projected_test_img)
recons_error = mean_squared_error(centered_test_img, reconstructed_test_img)

print(f"reconstruction error: {recons_error}")

# threshold = 7000

# if recons_error < threshold:
#     print("The image is a face.")
# else:
#     print("The image is not a face.")
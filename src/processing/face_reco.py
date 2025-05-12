import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import os
def upload_images():
    images=[]
    
    directory = 'C:/Users/rashe/Desktop/all/my files/Assignments/Image-Filters-And-Edge-Detection/training-synthetic'

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if filename.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png', '.bmp')):
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, (200, 200))  # Resize to 200x200
                images.append(img.flatten())
                
                cv.imshow('image', img)
                cv.waitKey(1)
                
    cv.destroyAllWindows() 
    np.save('data_set.npy', np.array(images))
    return np.array(images)

def PCA(X):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    small_cov = np.dot(X_centered, X_centered.T)  
    eigenvalues, eig_vecs_small = np.linalg.eigh(small_cov) 
    # Sort
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eig_vecs_small = eig_vecs_small[:, sorted_indices]

    eigenvectors = np.dot(X_centered.T, eig_vecs_small)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    sum_eigenvalues = np.sum(eigenvalues)
    eigenvalues = eigenvalues / sum_eigenvalues

    acumulated_eigenvalues = np.cumsum(eigenvalues)
    print(acumulated_eigenvalues[-1])
    for i in range(len(acumulated_eigenvalues)):
        if acumulated_eigenvalues[i] >= 0.99:
            print(f"Number of eigenvalues needed to explain 99% of the variance: {i+1} out of {len(eigenvectors)}")
            break
    eigenvectors = eigenvectors[:, :i+1]
    np.save('eigenvectors.npy', eigenvectors)
    return eigenvectors

def KNN(X, eigenvectors, img):
    mean = np.mean(X, axis=0)
    X_centered = X -mean
    X_reduced = np.dot(X_centered, eigenvectors)
    img_centered = img -mean
    img_reduced = np.dot(img_centered, eigenvectors)
    # distances = cosine_distances(X_reduced, img_reduced.reshape(1, -1)).flatten()
    # distances = np.linalg.norm(X_reduced - (img_reduced).reshape(1,-1), axis=1).flatten()
    distances = euclidean_distances(X_reduced, img_reduced.reshape(1,-1)).flatten()
    print("Distances:", len(distances))
    nearest_indices = np.argsort(distances)[:5]
    print("Nearest indices:", nearest_indices)
    return nearest_indices
    



def show_images(images):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))  
    for i, ax in enumerate(axes.flat):
        if(i==0):
            ax.set_title("Original Image")
        else:
            ax.set_title(f"Nearest Image {i}")
        ax.imshow(images[i], cmap='gray')  
        ax.axis('off')  


    plt.tight_layout()
    plt.show()

# X =upload_images()
X=np.load('data_set.npy')
# eigenvectors=PCA(X)
eigenvectors= np.load('eigenvectors.npy')#instead of calling PCA again, we load the eigenvectors calculated before
# img = cv.imread('C:/Users/rashe/Desktop/MIT-CBCL-facerec-database/test/0007_per170_-1426063360.pgm', cv.IMREAD_GRAYSCALE)
# img = cv.imread('C:/Users/rashe/Desktop/MIT-CBCL-facerec-database/test/0008_108.pgm', cv.IMREAD_GRAYSCALE)
img = cv.imread('C:/Users/rashe/Desktop/MIT-CBCL-facerec-database/test/0002_j_01229.pgm', cv.IMREAD_GRAYSCALE)

img = cv.resize(img, (200, 200)) 
KNN_indices = KNN(X, eigenvectors, img.flatten())
images = []
images.append(img)  
for i in KNN_indices:
    images.append(X[i].reshape(200, 200))  
show_images(images)



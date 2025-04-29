import numpy as np

def kmeans_segmentation(image, num_clusters=3, iterations = 10):
    # Reshape the image into (num_pixels, 3) array for RGB
    pixel_colors = image.reshape(-1, 3).astype(np.float64)
    labels = np.zeros((pixel_colors.shape[0],), dtype=np.int32)

    # Initialize centroids randomly by picking random pixel colors
    centroids = [pixel_colors[np.random.choice(pixel_colors.shape[0])] for _ in range(num_clusters)] # each centroid is a 3D vector for R, G, B

    for iteration in range(iterations):
        # Assign each pixel to the nearest centroid
        for idx, pixel in enumerate(pixel_colors):
            distances = [np.linalg.norm(pixel - centroid) for centroid in centroids] # distances between this pixel and all the centroids
            labels[idx] = np.argmin(distances)

        if iteration == iterations - 1:
            break

        # Update centroids
        new_centroids = []
        for cluster_idx in range(num_clusters):
            cluster_pixels = pixel_colors[labels == cluster_idx]
            if len(cluster_pixels) > 0:
                new_centroids.append(np.mean(cluster_pixels, axis=0))
            else:
                # Reinitialize empty cluster
                new_centroids.append(pixel_colors[np.random.choice(pixel_colors.shape[0])])

        centroids = new_centroids

    # Assign each pixel the color of its cluster's centroid
    segmented_pixels = np.zeros_like(pixel_colors)
    for cluster_idx in range(num_clusters):
        segmented_pixels[labels == cluster_idx] = centroids[cluster_idx]

    # Reshape back to original image shape
    segmented_image = segmented_pixels.reshape(image.shape)
    return segmented_image.astype(np.uint8)

#_________________________________________________________________________________________________________________________________
    # pixel_intensities = np.ravel(image).astype(np.float64)  # pixel_intensities is a 1D array of length equal to the number of pixels of the image, and the elements are each pixel's intensity
    # print(pixel_intensities.shape)
    # centroids = []
    # # initialize my centroids list
    # for i in range (num_clusters): 
    #     centroids.append(np.random.choice(pixel_intensities))  # appending a random pixel intensity as an initial centroid
    # print(centroids)
        
    # # initialize my cluster list
    # clusters = [[] for _ in range(num_clusters)]
    # for iteraion in range(iterations):
    #     # assign each pixel to a cluster by calculating distance between each pixel and each centroid
    #     for pixel in pixel_intensities:
    #         min_distance = np.inf
    #         for centroid_idx, centroid in enumerate(centroids):
    #             distance = np.abs(pixel - centroid)
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 cluster_centroid_idx = centroid_idx
    #         clusters[cluster_centroid_idx].append(pixel)
    #     # break the loop if we have reached the last iteration
    #     if iteraion == iterations - 1:
    #         break
        
    #     centroids = []
    #     # compute new centroids by computing each cluster's mean to be a new centroid
    #     for cluster in clusters:
    #         centroids.append(np.mean(cluster))
    #     clusters = [[] for _ in range(num_clusters)]
    
    # # loop on each pixel intensity in the image to assign the value of its cluster's centroid to it.
    # for pixel_idx, pixel in enumerate(pixel_intensities):
    #     for cluster_idx, cluster in enumerate(clusters):
    #         if pixel in cluster:
    #             pixel_intensities[pixel_idx] = centroids[cluster_idx]
    
    # segmented_image = pixel_intensities.reshape(image.shape)
    
    # return segmented_image.astype(np.uint8)
                
              
    
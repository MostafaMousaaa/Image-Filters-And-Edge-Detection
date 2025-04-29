import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
def get_neighbors(point, shape):
    
    x, y = point
    neighbors = []
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for dx, dy in dir:
        nx, ny = x + dx, y + dy
        if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
            neighbors.append((nx, ny)) 
    
    return neighbors

def region_growing(image, seed_point, threshold):
   
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    visited = np.zeros(image.shape, dtype=bool)
    region = np.zeros(image.shape, dtype=bool)

    seed_value = image[seed_point]
    
    queue = deque([seed_point])
    print(f'here {queue}')

    while queue:
        current_point = queue.popleft()
        x, y = current_point
        
        if visited[x, y]:
            continue
        
        visited[x, y] = True
        
        
        if abs(image[x, y] - seed_value) <= threshold:
            region[x, y] = 1
            
            neighbors = get_neighbors(current_point, image.shape)
            for n in neighbors:
                if not visited[n[0], n[1]]:
                    queue.append(n)

    return region

# np.random.seed(0)
# Original = cv2.imread('brain2.jpeg')
# image = cv2.imread('brain2.jpeg', cv2.IMREAD_GRAYSCALE)  
# print(image.shape)
# seed = (800, 750) 


# threshold = 0


# region = region_growing(image, seed, threshold)
# Original =  Original * region[:, :, np.newaxis]

# # نرسم النتائج
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# ax1.imshow(Original, cmap='gray')
# ax1.set_title('Original Image')
# ax1.plot(seed[1], seed[0], 'ro')  # نحدد نقطة البداية باللون الأحمر

# ax2.imshow(region, cmap='gray')
# ax2.set_title('Region Grown')

# plt.show()

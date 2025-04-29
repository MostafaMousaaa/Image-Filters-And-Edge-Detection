import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import deque
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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

def load_image():
    global Original, image, seed
    Tk().withdraw()  # Hide the root tkinter window
    file_path = askopenfilename(filetypes=[("Image files", "*.jpeg;*.jpg;*.png;*.bmp")])
    if file_path:
        Original = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        seed = (image.shape[0] // 2, image.shape[1] // 2)  # Reset seed to the center of the new image
        update(None)  # Update the display with the new image

# Load the initial image
Original = cv2.cvtColor(cv2.imread('brain2.jpeg'), cv2.COLOR_BGR2RGB)
image = cv2.imread('brain2.jpeg', cv2.IMREAD_GRAYSCALE)
seed = (800, 750)

# Initial threshold value
initial_threshold = 26

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Display the original image
ax1.imshow(Original)
ax1.set_title('Original Image')

# Display the initial region growing result
region = region_growing(image, seed, initial_threshold)
result = Original * region[:, :, np.newaxis]
region_plot = ax2.imshow(result)
ax2.set_title('Region Grown')

# Add a slider for the threshold
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
threshold_slider = Slider(ax_slider, 'Threshold', 0, 255, valinit=initial_threshold, valstep=1)

# Add a button to refresh the result
ax_button_refresh = plt.axes([0.8, 0.025, 0.1, 0.04])
refresh_button = Button(ax_button_refresh, 'Refresh')

# Add a button to load a new image
ax_button_load = plt.axes([0.65, 0.025, 0.1, 0.04])
load_button = Button(ax_button_load, 'Load Image')

# Update function for the slider and button
def update(val):
    global region_plot
    threshold = threshold_slider.val
    region = region_growing(image, seed, threshold)
    result = Original * region[:, :, np.newaxis]
    region_plot.set_data(result)
    ax1.imshow(Original)  # Update the original image display
    fig.canvas.draw_idle()

# Function to handle mouse clicks for setting the seed point
def on_click(event):
    global seed
    if event.inaxes == ax1:  # Check if the click is on the original image
        seed = (int(event.ydata), int(event.xdata))
        update(None)  # Update the region growing result

# Connect the slider, buttons, and mouse click event to their respective handlers
# threshold_slider.on_changed(update)
refresh_button.on_clicked(lambda event: update(None))
load_button.on_clicked(lambda event: load_image())
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

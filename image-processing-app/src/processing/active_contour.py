import numpy as np
import cv2

class GreedySnake:
    """
    Implementation of the Greedy Snake (Active Contour Model) algorithm.
    This implementation uses a greedy algorithm approach for evolving the contour.
    """
    
    def __init__(self, image, alpha=0.5, beta=0.5, gamma=0.1, max_iterations=100):
        """
        Initialize the active contour model.
        
        Parameters:
            image (numpy.ndarray): The image to apply the snake on
            alpha (float): Weight of continuity energy (elasticity)
            beta (float): Weight of curvature energy (stiffness)
            gamma (float): Weight of external energy (edge attraction)
            max_iterations (int): Maximum number of iterations
        """
        # Convert image to grayscale if it's color
        if len(image.shape) == 3:
            self.original_image = image.copy()
            self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.gray_image = image.copy()
        
        # Parameters
        self.alpha = alpha  # Continuity energy weight
        self.beta = beta    # Curvature energy weight
        self.gamma = gamma  # External energy weight
        self.max_iterations = max_iterations
        
        # Initialize contour points
        self.contour_points = []
        
        # For visualization
        self.evolution_history = []
        
        # Compute edge map (external force field)
        self._compute_edge_map()
    
    def _compute_edge_map(self):
        """Compute the edge map for external energy"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        
        # Calculate gradient magnitude using Sobel operators
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize gradient magnitude
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Invert gradient magnitude to create energy field (lower energy at edges)
        self.external_energy = 1 - gradient_magnitude
        
        # For visualization: create a color map
        edge_map_vis = (gradient_magnitude * 255).astype(np.uint8)
        self.edge_map_visualization = cv2.applyColorMap(edge_map_vis, cv2.COLORMAP_JET)
    
    def set_contour_points(self, points):
        """
        Set the initial contour points.
        
        Parameters:
            points (list): List of (x, y) tuples representing the initial contour
        """
        self.contour_points = np.array(points, dtype=np.float32)
        self.evolution_history = [self.contour_points.copy()]
    
    def _continuity_energy(self, prev_point, current_point, next_point):
        """
        Calculate continuity energy to maintain even spacing between points.
        
        Parameters:
            prev_point: Previous point in the contour
            current_point: Current point being evaluated
            next_point: Next point in the contour
            
        Returns:
            float: Continuity energy
        """
        # Calculate average distance between points
        d_mean = np.mean([np.linalg.norm(prev_point - current_point), 
                         np.linalg.norm(current_point - next_point)])
        
        # Calculate continuity energy (difference from mean distance)
        d_current = np.linalg.norm(prev_point - current_point)
        return (d_current - d_mean) ** 2
    
    def _curvature_energy(self, prev_point, current_point, next_point):
        """
        Calculate curvature energy to maintain smooth curves.
        
        Parameters:
            prev_point: Previous point in the contour
            current_point: Current point being evaluated
            next_point: Next point in the contour
            
        Returns:
            float: Curvature energy
        """
        # Calculate the second derivative approximation
        curvature = prev_point - 2 * current_point + next_point
        return np.linalg.norm(curvature) ** 2
    
    def _external_energy(self, point):
        """
        Calculate external energy from the edge map.
        
        Parameters:
            point: Point in the contour
            
        Returns:
            float: External energy
        """
        # Convert point to integer indices
        x, y = int(point[0]), int(point[1])
        
        # Ensure point is within image bounds
        if x < 0 or y < 0 or x >= self.external_energy.shape[1] or y >= self.external_energy.shape[0]:
            return 1.0  # High energy for out-of-bounds points
        
        # Return external energy at the point
        return self.external_energy[y, x]
    
    def _compute_total_energy(self, prev_point, current_point, next_point, new_point):
        """
        Compute the total energy for a potential new position.
        
        Parameters:
            prev_point: Previous point in the contour
            current_point: Current point being evaluated
            next_point: Next point in the contour
            new_point: Potential new position for current_point
            
        Returns:
            float: Total energy
        """
        # Calculate individual energy components
        continuity = self._continuity_energy(prev_point, new_point, next_point)
        curvature = self._curvature_energy(prev_point, new_point, next_point)
        external = self._external_energy(new_point)
        
        # Calculate weighted sum of energies
        return (self.alpha * continuity + 
                self.beta * curvature + 
                self.gamma * external)
    
    def evolve(self):
        """
        Evolve the contour using the greedy algorithm.
        
        Returns:
            numpy.ndarray: Final contour points
        """
        # Check if contour is initialized
        if len(self.contour_points) == 0:
            raise ValueError("Contour points must be initialized before evolving")
        
        # Number of contour points
        n = len(self.contour_points)
        
        # Define search neighborhood (8-connected neighborhood)
        neighborhood = np.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ])
        
        # Evolve the snake for max_iterations
        for iteration in range(self.max_iterations):
            # Flag to check if any point moved
            moved = False
            
            # New contour points
            new_contour = self.contour_points.copy()
            
            # For each point in the contour
            for i in range(n):
                # Get previous, current, and next points
                prev_point = self.contour_points[(i - 1) % n]
                current_point = self.contour_points[i]
                next_point = self.contour_points[(i + 1) % n]
                
                # Initialize minimum energy with current position energy
                min_energy = self._compute_total_energy(prev_point, current_point, next_point, current_point)
                best_point = current_point.copy()
                
                # Check each neighbor
                for neighbor in neighborhood:
                    # Calculate new potential position
                    new_point = current_point + neighbor
                    
                    # Compute energy at new position
                    energy = self._compute_total_energy(prev_point, current_point, next_point, new_point)
                    
                    # Update if energy is lower
                    if energy < min_energy:
                        min_energy = energy
                        best_point = new_point.copy()
                        moved = True
                
                # Update point in new contour
                new_contour[i] = best_point
            
            # Update contour points
            self.contour_points = new_contour.copy()
            
            # Save for visualization
            self.evolution_history.append(self.contour_points.copy())
            
            # Stop if no points moved
            if not moved:
                break
        
        return self.contour_points
    
    def get_visualization(self, show_edge_map=False, show_history=False):
        """
        Get a visualization of the contour.
        
        Parameters:
            show_edge_map (bool): Whether to show the edge map
            show_history (bool): Whether to show evolution history
            
        Returns:
            numpy.ndarray: Visualization image
        """
        # Create a copy of the original image for visualization
        vis_image = self.original_image.copy()
        
        # Show edge map if requested
        if show_edge_map:
            # Overlay edge map with transparency
            alpha = 0.7
            edge_overlay = cv2.addWeighted(
                vis_image, alpha, 
                self.edge_map_visualization, 1-alpha, 
                0
            )
            vis_image = edge_overlay
        
        # Draw the contour points
        if len(self.contour_points) > 0:
            # Convert points to integers for drawing
            contour_points_int = np.round(self.contour_points).astype(np.int32)
            
            # Draw history if requested
            if show_history and len(self.evolution_history) > 1:
                # Draw evolution with color gradient from red to green
                for i, hist_points in enumerate(self.evolution_history[:-1]):
                    hist_points_int = np.round(hist_points).astype(np.int32)
                    # Calculate color based on iteration (red → green)
                    progress = i / (len(self.evolution_history) - 1)
                    color = (
                        int((1 - progress) * 255),  # B: decrease
                        int(progress * 255),        # G: increase
                        0                           # R: fixed at 0
                    )
                    # Draw contour
                    cv2.polylines(vis_image, [hist_points_int], True, color, 1)
            
            # Draw final contour
            cv2.polylines(vis_image, [contour_points_int], True, (0, 0, 255), 2)
            
            # Draw points
            for point in contour_points_int:
                cv2.circle(vis_image, tuple(point), 3, (255, 0, 0), -1)
        
        return vis_image
    
    def calculate_metrics(self):
        """
        Calculate perimeter and area of the contour.
        
        Returns:
            tuple: (perimeter, area)
        """
        if len(self.contour_points) < 3:
            return 0, 0
        
        # Convert contour points to integer for OpenCV functions
        contour_points_int = np.round(self.contour_points).astype(np.int32)
        
        # Calculate perimeter
        perimeter = np.sum(np.sqrt(np.sum(np.diff(contour_points_int, axis=0)**2, axis=1)))
        
        # Calculate area
        area = 0.5 * np.abs(np.dot(contour_points_int[:, 0], np.roll(contour_points_int[:, 1], 1)) - np.dot(contour_points_int[:, 1], np.roll(contour_points_int[:, 0], 1)))
        
        return perimeter, area
    
    def get_chain_code(self):
        """
        Generate chain code representation of the contour.
        
        Returns:
            list: Chain code sequence
        """
        if len(self.contour_points) < 2:
            return []
        
        # Chain code direction mapping (8-connectivity):
        # 3 2 1
        # 4   0
        # 5 6 7
        
        chain_code = []
        # Convert points to integers
        points = np.round(self.contour_points).astype(np.int32)
        print(f'points: {points}')
        directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        direction_labels = ["R", "UR", "U", "UL", "L", "DL", "D", "DR"]
        # Calculate chain code for each pair of consecutive points
        for i in range(len(points)):
            current = points[i]
            next_point = points[(i + 1) % len(points)]
            
            # Calculate displacement vector
            dx = next_point[0] - current[0]
            dy = next_point[1] - current[1]
            
            # Determine direction code
            if (dx,dy) in directions:
                chain_code.append(direction_labels[directions.index((dx, dy))])
            
            
        print(chain_code)
        
        return chain_code

# Function to draw a contour on an image
def draw_contour(image, contour_points, color=(0, 0, 255), thickness=2):
    """
    Draw a contour on an image.
    
    Parameters:
        image (numpy.ndarray): Image to draw on
        contour_points (numpy.ndarray): Contour points
        color (tuple): Color in BGR format
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with contour
    """
    # Create a copy of the image
    result = image.copy()
    
    # Convert points to integers
    contour_points_int = np.round(contour_points).astype(np.int32)
    
    # Draw contour
    cv2.polylines(result, [contour_points_int], True, color, thickness)
    
    # Draw points
    for point in contour_points_int:
        cv2.circle(result, tuple(point), 3, (255, 0, 0), -1)
    
    return result

# Function to initialize a circular contour
def initialize_circular_contour(center, radius, num_points=20):
    """
    Initialize a circular contour.
    
    Parameters:
        center (tuple): Center of the circle (x, y)
        radius (float): Radius of the circle
        num_points (int): Number of points on the contour
        
    Returns:
        numpy.ndarray: Contour points
    """
    # Create points in a circle
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    # Combine x and y coordinates
    contour_points = np.column_stack((x, y))
    
    return contour_points

# Function to initialize a rectangular contour
def initialize_rectangular_contour(top_left, bottom_right, num_points=20):
    """
    Initialize a rectangular contour.
    
    Parameters:
        top_left (tuple): Top-left corner (x, y)
        bottom_right (tuple): Bottom-right corner (x, y)
        num_points (int): Total number of points on the contour
        
    Returns:
        numpy.ndarray: Contour points
    """
    # Extract coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Calculate points per side (approximately)
    points_per_side = max(2, num_points // 4)
    
    # Generate points along each side
    top = np.column_stack((
        np.linspace(x1, x2, points_per_side, endpoint=False),
        np.full(points_per_side, y1)
    ))
    
    right = np.column_stack((
        np.full(points_per_side, x2),
        np.linspace(y1, y2, points_per_side, endpoint=False)
    ))
    
    bottom = np.column_stack((
        np.linspace(x2, x1, points_per_side, endpoint=False),
        np.full(points_per_side, y2)
    ))
    
    left = np.column_stack((
        np.full(points_per_side, x1),
        np.linspace(y2, y1, points_per_side, endpoint=False)
    ))
    
    # Combine all sides
    contour_points = np.vstack((top, right, bottom, left))
    
    return contour_points

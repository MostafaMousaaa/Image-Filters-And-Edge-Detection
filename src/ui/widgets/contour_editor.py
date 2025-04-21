from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QSize
import numpy as np
import cv2

class ContourEditorWidget(QWidget):
    """
    Widget for initializing and editing contours on an image.
    """
    
    # Define signals
    contour_updated = pyqtSignal(list)  # Emitted when contour is updated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize variables
        self.image = None
        self.display_image = None
        self.pixmap = None
        self.contour_points = []
        self.is_drawing = False
        self.temp_point = None
        self.editing_mode = "manual"  # "manual", "circle", "rectangle"
        self.circle_center = None
        self.rect_start = None
        
        # Set up UI
        self.initUI()
    
    def initUI(self):
        # Set up the layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create label for displaying the image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(QSize(300, 300))
        self.image_label.setScaledContents(True)
        layout.addWidget(self.image_label)
        
        # Create a default blank pixmap
        blank_pixmap = QPixmap(600, 400)
        blank_pixmap.fill(Qt.GlobalColor.white)
        self.image_label.setPixmap(blank_pixmap)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
    
    def set_image(self, image):
        """
        Set the background image.
        
        Parameters:
            image (numpy.ndarray): Image in OpenCV format (BGR)
        """
        if image is None:
            return
        
        # Store the original image
        self.image = image.copy()
        
        # Convert OpenCV BGR to RGB for Qt
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # Grayscale image
            height, width = image.shape
            q_image = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
        
        # Create pixmap from QImage
        self.pixmap = QPixmap.fromImage(q_image)
        
        # Update display
        self.update_display()
    
    def set_contour_points(self, points):
        """
        Set contour points.
        
        Parameters:
            points (list): List of (x, y) tuples
        """
        self.contour_points = list(points)
        self.update_display()
        self.contour_updated.emit(self.contour_points)
    
    def clear_contour(self):
        """Clear all contour points."""
        self.contour_points = []
        self.temp_point = None
        self.is_drawing = False
        self.circle_center = None
        self.rect_start = None
        self.update_display()
        self.contour_updated.emit(self.contour_points)
    
    def set_editing_mode(self, mode):
        """
        Set the contour editing mode.
        
        Parameters:
            mode (str): "manual", "circle", or "rectangle"
        """
        self.editing_mode = mode
        self.clear_contour()
    
    def update_display(self):
        """Update the display with the current image and contour."""
        if self.pixmap is None:
            return
        
        # Create a copy of the pixmap for drawing
        display_pixmap = QPixmap(self.pixmap)
        painter = QPainter(display_pixmap)
        
        # Draw the contour
        if len(self.contour_points) > 0:
            # Set up pen for contour lines
            contour_pen = QPen(QColor(255, 0, 0))  # Red
            contour_pen.setWidth(2)
            painter.setPen(contour_pen)
            
            # Draw lines between points
            for i in range(len(self.contour_points)):
                p1 = QPoint(int(self.contour_points[i][0]), int(self.contour_points[i][1]))
                p2 = QPoint(int(self.contour_points[(i + 1) % len(self.contour_points)][0]), 
                           int(self.contour_points[(i + 1) % len(self.contour_points)][1]))
                painter.drawLine(p1, p2)
            
            # Draw points
            point_pen = QPen(QColor(0, 0, 255))  # Blue
            point_pen.setWidth(1)
            painter.setPen(point_pen)
            for point in self.contour_points:
                painter.drawEllipse(QPoint(int(point[0]), int(point[1])), 3, 3)
        
        # Draw temporary point or shape if we're in the middle of drawing
        if self.temp_point is not None:
            temp_pen = QPen(QColor(0, 255, 0))  # Green
            temp_pen.setWidth(2)
            painter.setPen(temp_pen)
            painter.drawEllipse(QPoint(int(self.temp_point[0]), int(self.temp_point[1])), 3, 3)
            
            # If we're drawing a shape, draw the preview
            if self.editing_mode == "circle" and self.circle_center is not None:
                # Calculate radius
                dx = self.temp_point[0] - self.circle_center[0]
                dy = self.temp_point[1] - self.circle_center[1]
                radius = np.sqrt(dx*dx + dy*dy)
                
                # Draw circle preview
                painter.drawEllipse(QPoint(int(self.circle_center[0]), int(self.circle_center[1])), 
                                  int(radius), int(radius))
                
            elif self.editing_mode == "rectangle" and self.rect_start is not None:
                # Draw rectangle preview
                x1, y1 = self.rect_start
                x2, y2 = self.temp_point
                painter.drawRect(int(min(x1, x2)), int(min(y1, y2)), 
                               int(abs(x2 - x1)), int(abs(y2 - y1)))
        
        # End painting
        painter.end()
        
        # Update the label
        self.image_label.setPixmap(display_pixmap)
    
    def mousePressEvent(self, event):
        if self.pixmap is None or not self.image_label.underMouse():
            return
        
        # Get position relative to the image label
        pos = self.image_label.mapFrom(self, event.pos())
        
        # Calculate scale factors
        scale_x = self.image_label.pixmap().width() / self.image_label.width()
        scale_y = self.image_label.pixmap().height() / self.image_label.height()
        
        # Apply scaling
        pos_x = pos.x() * scale_x
        pos_y = pos.y() * scale_y
        
        # Handle based on editing mode
        if self.editing_mode == "manual":
            if event.button() == Qt.MouseButton.LeftButton:
                # Start drawing or add a point
                self.is_drawing = True
                self.contour_points.append([pos_x, pos_y])
                self.update_display()
                self.contour_updated.emit(self.contour_points)
        
        elif self.editing_mode == "circle":
            if event.button() == Qt.MouseButton.LeftButton:
                if self.circle_center is None:
                    # Set center point
                    self.circle_center = [pos_x, pos_y]
                    self.is_drawing = True
                else:
                    # Calculate radius
                    dx = pos_x - self.circle_center[0]
                    dy = pos_y - self.circle_center[1]
                    radius = np.sqrt(dx*dx + dy*dy)
                    
                    # Generate circle contour
                    center = self.circle_center
                    num_points = 30  # Number of points in the circle
                    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
                    x_points = center[0] + radius * np.cos(angles)
                    y_points = center[1] + radius * np.sin(angles)
                    
                    # Set as contour points
                    self.contour_points = list(zip(x_points, y_points))
                    
                    # Reset drawing state
                    self.is_drawing = False
                    self.circle_center = None
                    self.temp_point = None
                    
                    # Update display
                    self.update_display()
                    self.contour_updated.emit(self.contour_points)
        
        elif self.editing_mode == "rectangle":
            if event.button() == Qt.MouseButton.LeftButton:
                if self.rect_start is None:
                    # Set start point
                    self.rect_start = [pos_x, pos_y]
                    self.is_drawing = True
                else:
                    # Calculate rectangle dimensions
                    x1, y1 = self.rect_start
                    x2, y2 = [pos_x, pos_y]
                    
                    # Generate rectangle contour (clockwise from top-left)
                    top_left = [min(x1, x2), min(y1, y2)]
                    top_right = [max(x1, x2), min(y1, y2)]
                    bottom_right = [max(x1, x2), max(y1, y2)]
                    bottom_left = [min(x1, x2), max(y1, y2)]
                    
                    # Create more points for smoother contour
                    num_points = 5  # Number of points per side
                    
                    # Generate points along each side
                    points = []
                    
                    # Top side
                    for i in range(num_points):
                        t = i / num_points
                        x = top_left[0] + t * (top_right[0] - top_left[0])
                        y = top_left[1]
                        points.append([x, y])
                    
                    # Right side
                    for i in range(num_points):
                        t = i / num_points
                        x = top_right[0]
                        y = top_right[1] + t * (bottom_right[1] - top_right[1])
                        points.append([x, y])
                    
                    # Bottom side
                    for i in range(num_points):
                        t = i / num_points
                        x = bottom_right[0] - t * (bottom_right[0] - bottom_left[0])
                        y = bottom_right[1]
                        points.append([x, y])
                    
                    # Left side
                    for i in range(num_points):
                        t = i / num_points
                        x = bottom_left[0]
                        y = bottom_left[1] - t * (bottom_left[1] - top_left[1])
                        points.append([x, y])
                    
                    # Set as contour points
                    self.contour_points = points
                    
                    # Reset drawing state
                    self.is_drawing = False
                    self.rect_start = None
                    self.temp_point = None
                    
                    # Update display
                    self.update_display()
                    self.contour_updated.emit(self.contour_points)
    
    def mouseMoveEvent(self, event):
        if self.pixmap is None or not self.image_label.underMouse():
            return
        
        # Get position relative to the image label
        pos = self.image_label.mapFrom(self, event.pos())
        
        # Calculate scale factors
        scale_x = self.image_label.pixmap().width() / self.image_label.width()
        scale_y = self.image_label.pixmap().height() / self.image_label.height()
        
        # Apply scaling
        pos_x = pos.x() * scale_x
        pos_y = pos.y() * scale_y
        
        # Update temporary point
        self.temp_point = [pos_x, pos_y]
        
        # Update display
        self.update_display()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False

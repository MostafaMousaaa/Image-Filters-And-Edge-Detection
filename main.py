# Assignment 2: Edge and boundary detection (Hough transform and SNAKE)
#  For given images (grayscale and color)/you may run on other images
#  A) Tasks to implement
#  For all given images; detect edges using Canny edge detector, detect lines, circles, 
# ellipsed located in these images (if any). Superimpose the detected shapes on the 
# images.
#  For given images; initialize the contour for a given object and evolve the Active 
# Contour Model (snake) using the greedy algorithm. Represent the output as chain 
# code and compute the perimeter and the area inside these contours.

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from src.ui.main_window import MainWindow
from src.ui.style import apply_stylesheet
from src.ui.splash_screen import SplashScreen
from PyQt6.QtCore import QTimer

def main():
    app = QApplication(sys.argv)
    
    app = apply_stylesheet(app)
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    splash.simulate_loading()
    
    # Initialize main window with support for both assignments
    main_window = MainWindow(support_edge_detection=True, support_active_contours=True)
    
    def finish_splash():
        if splash.progress >= 100:
            splash.finish(main_window)
            main_window.show()
    
    timer = QTimer()
    timer.timeout.connect(finish_splash)
    timer.start(100)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

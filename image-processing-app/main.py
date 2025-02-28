import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from src.ui.main_window import MainWindow
from src.ui.style import apply_stylesheet
from src.ui.splash_screen import SplashScreen
from PyQt6.QtCore import QTimer

# Import the compiled resources
# import src.ui.resources_rc

def main():
    # Create the application instance
    app = QApplication(sys.argv)
    
    # Apply custom stylesheet
    app = apply_stylesheet(app)
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Start loading simulation
    splash.simulate_loading()
    
    # Create main window but don't show it yet
    main_window = MainWindow()
    
    # When splash finishes, show main window
    def finish_splash():
        if splash.progress >= 100:
            splash.finish(main_window)
            main_window.show()
    
    # Check every 100ms if splash has finished
    timer = QTimer()
    timer.timeout.connect(finish_splash)
    timer.start(100)
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

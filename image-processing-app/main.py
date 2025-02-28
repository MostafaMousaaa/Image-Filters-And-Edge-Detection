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
    
    main_window = MainWindow()
    
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

import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.ui.style import apply_stylesheet

def main():
    app = QApplication(sys.argv)
    app = apply_stylesheet(app)
    
    # Initialize main window with support for both assignments
    main_window = MainWindow(support_edge_detection=True, support_active_contours=True)
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

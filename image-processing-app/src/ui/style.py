from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

def apply_stylesheet(app):
    """Apply a custom professional black and blue stylesheet to the application"""
    
    # Set the application style to Fusion which works well with custom styling
    app.setStyle("Fusion")
    
    # Create a custom dark palette with blue accents
    dark_palette = QPalette()
    
    # Base colors - dark theme
    dark_color = QColor(28, 28, 30)          # Nearly black
    darker_color = QColor(20, 20, 22)        # Darker black
    lighter_dark = QColor(44, 44, 46)        # Slightly lighter black
    
    # Accent colors - blue spectrum
    accent_color = QColor(10, 132, 255)      # Bright blue
    accent_light = QColor(94, 174, 255)      # Lighter blue
    accent_dark = QColor(0, 88, 208)         # Darker blue
    
    # Text colors
    text_color = QColor(242, 242, 247)       # Very light gray/white
    secondary_text = QColor(174, 174, 178)   # Lighter gray
    disabled_text = QColor(99, 99, 102)      # Medium gray
    
    # Set up the palette
    dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
    dark_palette.setColor(QPalette.ColorRole.WindowText, text_color)
    dark_palette.setColor(QPalette.ColorRole.Base, darker_color)
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, lighter_dark)
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, darker_color)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, text_color)
    dark_palette.setColor(QPalette.ColorRole.Text, text_color)
    dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ButtonText, text_color)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Link, accent_light)
    dark_palette.setColor(QPalette.ColorRole.Highlight, accent_color)
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    
    # Disabled states
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, lighter_dark)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, disabled_text)
    
    # Apply the palette
    app.setPalette(dark_palette)
    
    # Additional stylesheet for fine-grained control
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1c1c1e;
        }
        
        QMenuBar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #1a1a1a, stop:1 #0a0a0a);
            color: #f2f2f7;
            border-bottom: 1px solid #0a84ff;
        }
        
        QMenuBar::item {
            background: transparent;
            padding: 6px 10px;
            margin: 2px;
            border-radius: 4px;
        }
        
        QMenuBar::item:selected {
            background: rgba(10, 132, 255, 0.6);
        }
        
        QMenuBar::item:pressed {
            background: #0a84ff;
        }
        
        QMenu {
            background-color: #1c1c1e;
            border: 1px solid #0a84ff;
            border-radius: 3px;
            padding: 3px;
        }
        
        QMenu::item {
            padding: 6px 25px 6px 25px;
            border-radius: 3px;
        }
        
        QMenu::item:selected {
            background-color: rgba(10, 132, 255, 0.8);
            color: white;
        }
        
        QToolBar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #2c2c2e, stop:1 #1c1c1e);
            border-bottom: 1px solid #0a84ff;
            spacing: 3px;
            padding: 3px;
        }
        
        QToolButton {
            background: transparent;
            border-radius: 4px;
            border: 1px solid transparent;
            padding: 5px;
        }
        
        QToolButton:hover {
            background-color: rgba(10, 132, 255, 0.3);
            border: 1px solid rgba(10, 132, 255, 0.6);
        }
        
        QToolButton:pressed {
            background-color: rgba(10, 132, 255, 0.8);
        }
        
        QTabWidget {
            border: none;
        }
        
        QTabWidget::pane {
            border: 1px solid #444446;
            border-radius: 4px;
            top: -1px;
            background: #1c1c1e;
        }
        
        QTabBar::tab {
            background: #2c2c2e;
            color: #aeaeb2;
            border: 1px solid #444446;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 8px 12px;
            min-width: 80px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #0a84ff, stop:1 #0058d0);
            color: white;
        }
        
        QTabBar::tab:!selected:hover {
            background: #3c3c3e;
            border: 1px solid #0a84ff;
            border-bottom: none;
        }
        
        QStatusBar {
            background-color: #1c1c1e;
            color: #aeaeb2;
            border-top: 1px solid #0a84ff;
            padding: 2px;
        }
        
        QGroupBox {
            border: 1px solid #444446;
            border-radius: 6px;
            margin-top: 16px;
            font-weight: bold;
            background-color: #2c2c2e;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: #0a84ff;
        }
        
        QPushButton {
            background-color: #0a84ff;
            color: white;
            border-radius: 6px;
            padding: 8px 16px;
            border: none;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #5eaeff;
        }
        
        QPushButton:pressed {
            background-color: #0058d0;
        }
        
        QPushButton:disabled {
            background-color: #2c2c2e;
            color: #636366;
            border: 1px solid #444446;
        }
        
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #444446;
            border-radius: 4px;
            padding: 3px;
            background-color: #2c2c2e;
            color: #f2f2f7;
            selection-background-color: #0a84ff;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #0a84ff;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #444446;
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
        }
        
        QComboBox::down-arrow {
            image: url(icons/down-arrow.png);
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #444446;
            height: 4px;
            background: #2c2c2e;
            margin: 0px;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                      stop:0 #5eaeff, stop:1 #0a84ff);
            border: 1px solid #0a84ff;
            width: 18px;
            height: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #5eaeff;
            border: 1px solid #5eaeff;
        }
        
        QSplitter::handle {
            background-color: #0a84ff;
            height: 1px;
        }
        
        QSplitter::handle:horizontal {
            width: 2px;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #2c2c2e;
            width: 10px;
            margin: 16px 0 16px 0;
            border-radius: 4px;
        }

        QScrollBar::handle:vertical {
            background: #0a84ff;
            min-height: 20px;
            border-radius: 4px;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical,
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        
        QScrollBar:horizontal {
            border: none;
            background: #2c2c2e;
            height: 10px;
            margin: 0px 16px 0 16px;
            border-radius: 4px;
        }

        QScrollBar::handle:horizontal {
            background: #0a84ff;
            min-width: 20px;
            border-radius: 4px;
        }

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
        QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none;
        }
        
        /* Custom styling for image frames */
        QFrame#imageFrame, QFrame#histFrame {
            background-color: #14141a;
            border: 1px solid #0a84ff;
            border-radius: 8px;
        }
        
        /* Edge Detection and Active Contour Panels */
        QGroupBox#paramGroupBox, QGroupBox#actionGroupBox, QGroupBox#metricsGroupBox {
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-top: 1ex;
            font-weight: bold;
        }
        
        QGroupBox#paramGroupBox::title, QGroupBox#actionGroupBox::title, QGroupBox#metricsGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        
        QLabel#paramLabel {
            min-width: 120px;
        }
        
        QLabel#valueLabel {
            min-width: 40px;
            font-weight: bold;
        }
        
        QLabel#metricLabel {
            font-weight: bold;
        }
        
        QLabel#metricValue {
            font-weight: bold;
            color: #0066cc;
        }
        
        QLabel#statusLabel {
            font-weight: bold;
            color: #005500;
            padding: 5px;
        }
        
        #edgeDetectionPanel QPushButton#actionButton, 
        #activeContourPanel QPushButton#actionButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            min-height: 30px;
        }
        
        #edgeDetectionPanel QPushButton#actionButton:hover, 
        #activeContourPanel QPushButton#actionButton:hover {
            background-color: #45a049;
        }
        
        #edgeDetectionPanel QPushButton#actionButton:pressed, 
        #activeContourPanel QPushButton#actionButton:pressed {
            background-color: #3c8c40;
        }
        
        QFrame#statusFrame, QFrame#resultsFrame {
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
            margin-top: 5px;
            padding: 5px;
        }
        
        QSlider#paramSlider::groove:horizontal {
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
        }
        
        QSlider#paramSlider::handle:horizontal {
            background: #4CAF50;
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        
        QDoubleSpinBox#paramSpinBox, QSpinBox#paramSpinBox {
            padding: 3px;
            min-width: 70px;
        }
        
        QTabWidget::pane {
            border: 1px solid #ccc;
            border-radius: 3px;
            top: -1px;
        }
        
        QTabBar::tab {
            background: #f0f0f0;
            border: 1px solid #ccc;
            padding: 5px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background: white;
            border-bottom-color: white;
        }
        
        QDockWidget#edgeDockWidget, QDockWidget#contourDockWidget {
            font-weight: bold;
        }
    """)
    
    # Set modern font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    return app

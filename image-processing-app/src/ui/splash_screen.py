from PyQt6.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QLabel, QWidget
from PyQt6.QtGui import QPixmap, QFont, QPainter, QColor, QBrush, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, QSize

class SplashScreen(QSplashScreen):
    def __init__(self):
        # Create a pixmap for the splash screen with black and blue gradient
        pixmap = QPixmap(600, 400)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        # Create a painter to draw on the pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0, QColor(28, 28, 30))      # Dark at top
        gradient.setColorAt(1, QColor(10, 10, 12))      # Darker at bottom
        painter.fillRect(0, 0, 600, 400, QBrush(gradient))
        
        # Draw blue accent lines
        painter.setPen(QColor(10, 132, 255))
        painter.drawLine(0, 2, 600, 2)
        painter.drawLine(0, 398, 600, 398)
        
        # End painting
        painter.end()
        
        # Initialize splash screen with the pixmap
        super().__init__(pixmap)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        
        # Add content to the splash screen
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # App title
        self.title_label = QLabel("Computer Vision Assignment 1\n Image Filters And Edge Detection\n")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            color: #ffffff;
            font-weight: bold;
            font-size: 28px;
        """)
        layout.addWidget(self.title_label)
        
        # Spacer
        layout.addSpacing(40)
        
        # Loading label
        self.loading_label = QLabel("Loading Application...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("""
            color: #0a84ff;
            font-size: 16px;
        """)
        layout.addWidget(self.loading_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444446;
                border-radius: 5px;
                background-color: #2c2c2e;
                color: white;
                text-align: center;
                height: 25px;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                               stop:0 #0058d0, stop:1 #0a84ff);
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            color: #aeaeb2;
            font-size: 12px;
        """)
        layout.addWidget(self.status_label)
        
        # Version label
        self.version_label = QLabel("Version 1.0")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.version_label.setStyleSheet("""
            color: #636366;
            font-size: 10px;
        """)
        layout.addSpacing(50)
        layout.addWidget(self.version_label)
        
        # Create a container widget to hold the layout
        container = QWidget(self)
        container.setLayout(layout)
        
        # Position the container on the splash screen
        container.setGeometry(0, 0, 600, 400)
        
        # Initialize progress value
        self.progress = 0
        
    def simulate_loading(self):
        """Simulate a loading process with steps"""
        self.loading_steps = [
            "Initializing application...",
            "Loading resources...",
            "Setting up UI components...",
            "Configuring image processors...",
            "Initializing filters...",
            "Setting up edge detectors...",
            "Preparing histogram analyzers...",
            "Loading frequency domain processors...",
            "Configuring hybrid image tools...",
            "Starting application..."
        ]
        
        # Create a timer to update progress
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(150)  # Update every 150ms
        
    def update_progress(self):
        """Update the progress bar and status message"""
        if self.progress < 100:
            # Increment progress
            self.progress += 2
            self.progress_bar.setValue(self.progress)
            
            # Update status message based on progress
            step_index = min(int(self.progress / 10), len(self.loading_steps) - 1)
            self.status_label.setText(self.loading_steps[step_index])
            
        else:
            # Loading complete
            self.timer.stop()
            self.status_label.setText("Ready")

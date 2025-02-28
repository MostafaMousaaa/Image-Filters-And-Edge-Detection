# Image Processing Application

This project is an image processing application built using PyQt6. It implements various image processing techniques including noise addition, filtering, edge detection, histogram generation, thresholding, and more. The application provides a user-friendly interface for performing these tasks on images.

## Project Structure

```
image-processing-app
├── src
│   ├── main.py                     # Entry point of the application
│   ├── ui                          # UI components
│   │   ├── __init__.py             # Marks the ui directory as a package
│   │   ├── main_window.py           # Main application window setup
│   │   └── widgets                  # Custom widgets for the UI
│   │       ├── __init__.py          # Marks the widgets directory as a package
│   │       ├── histogram_widget.py   # Widget for displaying histograms
│   │       ├── image_display_widget.py # Widget for displaying images
│   │       └── filter_controls_widget.py # Widget for filter controls
│   ├── processing                   # Image processing modules
│   │   ├── __init__.py              # Marks the processing directory as a package
│   │   ├── noise.py                 # Functions for adding noise to images
│   │   ├── filters.py               # Implementations of low-pass filters
│   │   ├── edge_detection.py         # Functions for edge detection
│   │   ├── histogram.py              # Functions for histogram generation
│   │   ├── thresholding.py           # Functions for thresholding methods
│   │   ├── frequency_domain.py       # Functions for frequency domain filters
│   │   └── hybrid_images.py          # Functions for creating hybrid images
│   └── utils                        # Utility functions
│       ├── __init__.py              # Marks the utils directory as a package
│       └── image_utils.py           # Utility functions for image handling
├── resources
│   └── images
│       └── sample.jpg               # Sample image for testing
├── tests
│   ├── __init__.py                  # Marks the tests directory as a package
│   ├── test_filters.py              # Unit tests for filters
│   └── test_edge_detection.py        # Unit tests for edge detection
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd image-processing-app
   ```

2. **Install dependencies**:
   Make sure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Execute the following command to start the application:
   ```
   python src/main.py
   ```

## Usage

- Load an image using the provided interface.
- Apply various noise types to the image.
- Use different filters to process the image.
- Detect edges using multiple methods.
- View histograms and distribution curves.
- Apply thresholding techniques.
- Experiment with frequency domain filters and hybrid images.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
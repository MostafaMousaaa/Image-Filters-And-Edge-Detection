from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QScrollArea
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class ChainCodeDisplay(QWidget):
    """
    Widget for displaying chain code representation of a contour.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        # Set up the layout
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Chain Code Representation")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        description = (
            "Chain code represents the contour as a sequence of directional codes (0-7):\n"
            "0: Right, 1: Up-Right, 2: Up, 3: Up-Left,\n"
            "4: Left, 5: Down-Left, 6: Down, 7: Down-Right"
        )
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Create text edit for chain code display
        self.code_display = QTextEdit()
        self.code_display.setReadOnly(True)
        self.code_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        
        # Set monospaced font for better code display
        code_font = QFont("Courier New", 10)
        self.code_display.setFont(code_font)
        
        # Add to layout inside a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.code_display)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
    
    def set_chain_code(self, chain_code):
        """
        Set the chain code to display.
        
        Parameters:
            chain_code (list): List of chain code values (0-7)
        """
        if not chain_code:
            self.code_display.setText("No chain code available.")
            return
        
        # Format the chain code for display
        # Show in groups of 10 for readability
        formatted_code = ""
        for i in range(0, len(chain_code), 10):
            group = chain_code[i:i+10]
            formatted_code += " ".join(str(code) for code in group)
            if i + 10 < len(chain_code):
                formatted_code += "\n"
        
        # Include statistics
        stats = (
            f"Chain Code Length: {len(chain_code)}\n"
            f"Chain Code Direction Frequencies:\n"
        )
        
        # Count frequencies of each direction
        for direction in range(8):
            count = chain_code.count(direction)
            percentage = (count / len(chain_code)) * 100 if chain_code else 0
            stats += f"  Direction {direction}: {count} occurrences ({percentage:.1f}%)\n"
        
        # Set the text
        self.code_display.setText(f"{stats}\nChain Code:\n{formatted_code}")

import os

class IconPaths:
    """Helper class to manage icon paths"""
    
    _base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resources')
    
    @classmethod
    def get_icon(cls, name):
        """Get the absolute path for an icon by name"""
        return os.path.join(cls._base_path, f"{name}.png")

    # Define constants for commonly used icons
    OPEN = property(lambda self: self.get_icon("open"))
    SAVE = property(lambda self: self.get_icon("save"))
    RESET = property(lambda self: self.get_icon("reset"))
    GRAYSCALE = property(lambda self: self.get_icon("grayscale"))
    EQUALIZE = property(lambda self: self.get_icon("equalize"))
    NORMALIZE = property(lambda self: self.get_icon("normalize"))
    APP_ICON = property(lambda self: self.get_icon("app_icon"))
    NOISE = property(lambda self: self.get_icon("noise"))
    FILTER = property(lambda self: self.get_icon("filter"))
    EDGE = property(lambda self: self.get_icon("edge"))
    THRESHOLD = property(lambda self: self.get_icon("threshold"))
    FREQUENCY = property(lambda self: self.get_icon("frequency"))
    HYBRID = property(lambda self: self.get_icon("hybrid"))
    MERGE = property(lambda self: self.get_icon("merge"))
    IMAGE = property(lambda self: self.get_icon("image"))

# Create a singleton instance
icons = IconPaths()

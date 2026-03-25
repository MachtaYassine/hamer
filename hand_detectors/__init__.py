from .base import HandDetection, HandDetector
from .vitpose_detect import ViTPoseHandDetector

def create_hand_detector(**kwargs) -> HandDetector:
    return ViTPoseHandDetector(**kwargs)

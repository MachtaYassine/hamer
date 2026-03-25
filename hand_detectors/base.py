"""Base interface for hand detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HandDetection:
    """A single detected hand in a frame."""
    bbox: np.ndarray             # (4,) [x_min, y_min, x_max, y_max] pixel coords
    is_right: bool               # True = right hand, False = left hand
    confidence: float            # detection confidence [0, 1]
    keypoints: np.ndarray = field(default=None)  # (21, 3) [x, y, conf] or None


class HandDetector(ABC):
    """Abstract base for hand bbox detectors.

    All detectors must produce the same output: a list of HandDetection per frame.
    This feeds into ViTDetDataset → HaMeR MANO regression unchanged.
    """

    @abstractmethod
    def detect_hands(self, img_bgr: np.ndarray) -> list[HandDetection]:
        """Detect hands in a single BGR frame.

        Returns list of HandDetection (may be empty if no hands found).
        """
        ...

    def close(self):
        """Release resources. Override if needed."""
        pass

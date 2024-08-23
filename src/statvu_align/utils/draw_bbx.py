import cv2
import numpy as np
from typing import List

def draw_bbx(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Draw a bounding-box around an img: `frame` and return the results.
    """

    frame = frame.copy()
    x1, y1, x2, y2 = xyxy.astype(int)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
    cv2.putText(
                    frame,
                    "Time Remaining)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    4,
                )
    return frame

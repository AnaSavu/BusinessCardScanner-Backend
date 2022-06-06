import logging

import cv2
import numpy as np

class CannyEdgeDetector:
    def __init__(self, image):
        self.__logger = logging.getLogger("Canny Edge Detector")
        self.__image = image

    def get_canny_image(self, sigma=0.33):
        v = np.median(self.__image)

        if v > 191:
            lower = int(max(0, (1 - 2 * sigma) * (255 - v)))
            upper = int(max(85, (1 + 2 * sigma) * (255 - v)))

        elif v > 127:
            lower = int(max(0, (1 - sigma) * (255 - v)))
            upper = int(max(255, (1 + sigma) * (255 - v)))

        elif v < 63:
            lower = int(max(0, (1 - 2 * sigma) * v))
            upper = int(max(85, (1 + 2 * sigma) * v))

        else:
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))

        return cv2.Canny(self.__image, lower, upper, apertureSize=3)

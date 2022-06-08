import logging

import cv2
import numpy as np

class CannyEdgeDetector:
    def __init__(self, image):
        self.__image = image

    def get_canny_image(self, sigma=0.33):
        median = np.median(self.__image)

        if median > 191:
            lower_threshold = int(max(0, (1 - 2 * sigma) * (255 - median)))
            upper_threshold = int(max(85, (1 + 2 * sigma) * (255 - median)))

        elif median > 127:
            lower_threshold = int(max(0, (1 - sigma) * (255 - median)))
            upper_threshold = int(max(255, (1 + sigma) * (255 - median)))

        elif median < 63:
            lower_threshold = int(max(0, (1 - 2 * sigma) * median))
            upper_threshold = int(max(85, (1 + 2 * sigma) * median))

        else:
            lower_threshold = int(max(0, (1.0 - sigma) * median))
            upper_threshold = int(min(255, (1.0 + sigma) * median))

        return cv2.Canny(self.__image, lower_threshold, upper_threshold, apertureSize=3)

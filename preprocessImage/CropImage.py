import logging
import cv2
import numpy as np


class CropImage:
    def __init__(self, image):
        self.__image = image

    def __four_point_transform(self, points):
        rectangle = self.__order_points(points)
        (top_left, top_right, bottom_right, bottom_left) = rectangle

        widthA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        widthB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
        heightB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        distance = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rectangle, distance)
        warped = cv2.warpPerspective(self.__image, M, (maxWidth, maxHeight))

        return warped

    def __order_points(self, points):
        points = points.reshape(4, 2)
        rectangle = np.zeros((4, 2), dtype="float32")

        sum = points.sum(axis=1)
        rectangle[0] = points[np.argmin(sum)]
        rectangle[2] = points[np.argmax(sum)]

        difference = np.diff(points, axis=1)
        rectangle[1] = points[np.argmin(difference)]
        rectangle[3] = points[np.argmax(difference)]

        return rectangle

    def crop_image_into_business_card(self, approx):
        distance = 0
        if approx is not None and len(approx) == 4:
            approx_contour = np.float32(approx)
            distance = self.__four_point_transform(approx_contour)
        return distance
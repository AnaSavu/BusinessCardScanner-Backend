import logging

import cv2
import imutils
import numpy as np


class ContourImage:
    def __init__(self, image):
        self.__image = image

    def get_contours(self, resized_image):
        contours, hierarchy = cv2.findContours(self.__image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = []

        for contour in contours:
            hull = cv2.convexHull(contour) # a concave shape, will be approximated to a convex boundary that most tightly encloses it
            simplified_contours.append(cv2.approxPolyDP(hull,
                                                        0.001 * cv2.arcLength(hull, True), True)) # a shape is approximated to a another shape consisitng of a lesser number of vertices
        simplified_contours = np.array(simplified_contours)

        biggest_n, approx_contour = self.__get_biggest_rectangle(simplified_contours, resized_image.size)
        image_copy = resized_image.copy()

        threshold = cv2.drawContours(image_copy, simplified_contours, biggest_n, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        self.__image = threshold
        return threshold, image_copy, approx_contour

    def __get_biggest_rectangle(self, contours, min_area):
        max_area = 0
        biggest_n = 0
        approx_contour = None
        for n, i in enumerate(contours):
            area = cv2.contourArea(i)

            if area > min_area / 10:
                approx = cv2.approxPolyDP(i, 0.02 * (cv2.arcLength(i, True)), True)
                if area > max_area and len(approx) == 4:
                    max_area = area
                    biggest_n = n
                    approx_contour = approx

        return biggest_n, approx_contour

import logging

import cv2
import imutils
import numpy as np


class ContourImage:
    def __init__(self, image):
        self.__logger = logging.getLogger("Contour")
        self.__image = image
        # self.__contours = None

    def get_contours(self, resized_image):
        contours, hierarchy = cv2.findContours(self.__image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = []

        for cnt in contours:
            hull = cv2.convexHull(cnt)
            simplified_contours.append(cv2.approxPolyDP(hull,
                                                        0.001 * cv2.arcLength(hull, True), True))
        simplified_contours = np.array(simplified_contours)

        biggest_n, approx_contour = self. __get_biggest_rectangle(simplified_contours, resized_image.size)
        image_copy = resized_image.copy()

        threshold = cv2.drawContours(image_copy, simplified_contours, biggest_n, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        self.__image = threshold
        return threshold, image_copy, approx_contour

        # cnts, hierarchy = cv2.findContours(self.__image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        # # loop over the contours
        # for c in cnts:
        #     # approximate the contour
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #     # if our approximated contour has four points, then we
        #     # can assume that we have found our screen
        #     if len(approx) == 4:
        #         screenCnt = approx
        #         break
        #
        # image_copy = resized_image.copy()
        #
        # threshold = cv2.drawContours(image_copy, [screenCnt], -1, (0, 255, 0), 2)
        # return threshold, image_copy, screenCnt

    def __get_biggest_rectangle(self, contours, min_area):
        biggest = None
        max_area = 0
        biggest_n = 0
        approx_contour = None
        for n, i in enumerate(contours):
            area = cv2.contourArea(i)

            if area > min_area / 10:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
                    biggest_n = n
                    approx_contour = approx

        return biggest_n, approx_contour

    # def getContours(self):
    #     return self.__contours
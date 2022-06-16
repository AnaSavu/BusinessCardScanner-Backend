import logging

import cv2

from preprocessImage.CannyEdgeDetector import CannyEdgeDetector
from preprocessImage.ContourImage import ContourImage
from preprocessImage.CropImage import CropImage
from preprocessImage.EditImage import EditImage


class PreProcessImageWrapper:
    def __init__(self):
        pass

    def main_preporcess_image_wrapper(self):
        image = cv2.imread("test.jpeg")

        resized_image = EditImage(image).resize_image()
        edit_image = EditImage(image).edit_image()
        canny_image = CannyEdgeDetector(edit_image).get_canny_image()

        contour_object = ContourImage(canny_image)
        th, im, approx = contour_object.get_contours(resized_image)

        crop_object = CropImage(th)
        cropped_image = crop_object.crop_image_into_business_card(approx)
        try:
            if cropped_image == 0:
                cv2.imwrite("output.jpeg", image)
        except:
            cv2.imwrite("output.jpeg", cropped_image)
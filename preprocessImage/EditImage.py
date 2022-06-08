import logging
import cv2
import numpy as np

class EditImage:
    def __init__(self, image):
        self.__image = image

    def __convert_grayscale_and_blur(self):
        grayscale_image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        # image_y = np.zeros(yuv_image.shape[0:2], np.uint8)
        # image_y[:, :] = yuv_image[:, :, 0]
        blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        return blur_image

    # def _equalizeHistogram(self):
    #     return cv2.equalizeHist(self.__image)
    #
    # def _binaryThreshold(self):
    #     ret, thresh = cv2.threshold(self._convertImageAndBlur(), 150, 255, cv2.THRESH_BINARY)
    #     return thresh

    def resize_image(self):
        self.__image = cv2.resize(self.__image, (0, 0), fx=0.5, fy=0.5)
        return self.__image

    def __contrast_and_brightness(self, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness

            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buffer = cv2.addWeighted(self.__image, alpha_b, self.__image, 0, gamma_b)
        else:
            buffer = self.__image.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buffer = cv2.addWeighted(buffer, alpha_c, buffer, 0, gamma_c)

        return buffer

    def edit_image(self):
        self.__image = self.__contrast_and_brightness(-64, 64)
        self.__image = self.__convert_grayscale_and_blur()
        # self.__image =  self._equalizeHistogram()

        return self.resize_image()


    def getEditedImage(self):
        return self.__image
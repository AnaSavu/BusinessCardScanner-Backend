import logging
import cv2
import numpy as np

class EditImage:
    def __init__(self, image):
        self.__logger = logging.getLogger("Edit Image")
        self.__image = image

    def _convertImageAndBlur(self):
        self.__logger.info("Gray scale image and apply Gaussian filter for smoothening")
        yuv_image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        # image_y = np.zeros(yuv_image.shape[0:2], np.uint8)
        # image_y[:, :] = yuv_image[:, :, 0]
        blur_image = cv2.GaussianBlur(yuv_image, (5, 5), 0)
        return blur_image

    def _equalizeHistogram(self):
        return cv2.equalizeHist(self.__image)

    def _binaryThreshold(self):
        ret, thresh = cv2.threshold(self._convertImageAndBlur(), 150, 255, cv2.THRESH_BINARY)
        return thresh

    def _resizeImage(self):
        self.__logger.info("Resize the image by half")
        self.__logger.info(self.__image.shape)
        self.__image = cv2.resize(self.__image, (0, 0), fx=0.5, fy=0.5)
        self.__logger.info(self.__image.shape)
        return self.__image

    def contrast_and_brightness(self, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness

            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(self.__image, alpha_b, self.__image, 0, gamma_b)
        else:
            buf = self.__image.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def _editImage(self):
        self.__image = self.contrast_and_brightness(-64, 64)
        self.__image = self._convertImageAndBlur()
        # self.__image =  self._equalizeHistogram()

        return self._resizeImage()


    def getEditedImage(self):
        return self.__image
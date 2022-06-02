import base64

class ImageConvertor:
    def __init__(self, image):
        self.__image = image

    def toBase64(self):
        with open("output.jpeg", "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read())
        return encoded_string


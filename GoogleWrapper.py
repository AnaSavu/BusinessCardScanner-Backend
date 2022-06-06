from google.HttpRequest import HttpRequest
from google.ImageConvertor import ImageConvertor


class GoogleWrapper:
    def __init__(self):
        pass

    def main_google_apis_wrapper(self):
        string_image = ImageConvertor("output.jpeg").toBase64()
        return HttpRequest(string_image).get_response_from_google_nlp()
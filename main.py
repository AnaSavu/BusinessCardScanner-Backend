import base64
import cv2
import uvicorn

from fastapi import FastAPI, Form, File
from pydantic import BaseModel

from GoogleWrapper import GoogleWrapper
from PreProcessImageWrapper import PreProcessImageWrapper
from google.HttpRequest import HttpRequest
from google.ImageConvertor import ImageConvertor
from preprocessImage.CannyEdgeDetector import CannyEdgeDetector
from preprocessImage.CropImage import CropImage
from preprocessImage.EditImage import EditImage

app = FastAPI()

class Image(BaseModel):
    base64str: str

@app.get("/")
async def root():
    return {"greeting":"Hello world"}

@app.post("/image")
async def getImage(image: Image):
    decodeImage = open("test.jpeg", "wb")
    decodeImage.write(base64.b64decode(image.base64str))

    PreProcessImageWrapper().main_preporcess_image_wrapper()
    google_result = GoogleWrapper().main_google_apis_wrapper()

    return google_result


# def mainImageWrapper():
#     image = cv2.imread("test.jpeg")
#
#     resized_image = EditImage(image).resize_image()
#     edit_image = EditImage(image).edit_image()
#     canny_image = CannyEdgeDetector(edit_image).get_canny_image()
#
#     contour_object = Contour(canny_image)
#     th, im, approx = contour_object.get_contours(resized_image)
#
#     crop_object = CropImage(th)
#     cropped_image = crop_object.crop_image_into_bc(approx)
#     cv2.imwrite("output.jpeg", cropped_image)

# def mainOCRWrapper():
#     string_image = ImageConvertor("output.jpeg").toBase64()
#     return HttpRequest(string_image).getHttpResponseFromNLP()
#
# if __name__ == "__main__":
#     uvicorn.run('main:app', host="0.0.0.0", port=8080, reload=True)

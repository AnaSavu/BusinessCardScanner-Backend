import base64
import cv2
import uvicorn

from fastapi import FastAPI, Form, File
from pydantic import BaseModel

from ocr.HttpRequest import HttpRequest
from ocr.ImageConvertor import ImageConvertor
from preprocessImage.CannyEdgeDetector import CannyEdgeDetector
from preprocessImage.ContourImage import Contour
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

    mainImageWrapper()
    ocr_result = mainOCRWrapper()

    return ocr_result


def mainImageWrapper():
    image = cv2.imread("test.jpeg")

    resized_image = EditImage(image)._resizeImage()
    edit_image = EditImage(image)._editImage()
    canny_image = CannyEdgeDetector(edit_image)._applyCannyAlgorithm()

    contour_object = Contour(canny_image)
    th, im, approx = contour_object.findContours(resized_image)

    crop_object = CropImage(th)
    croppedImage = crop_object.cropImageIntoBC(approx)
    cv2.imwrite("output.jpeg", croppedImage)

def mainOCRWrapper():
    string_image = ImageConvertor("output.jpeg").toBase64()
    return HttpRequest(string_image).getHttpResponseFromNLP()
#
# if __name__ == "__main__":
#     uvicorn.run('main:app', host="0.0.0.0", port=8080, reload=True)

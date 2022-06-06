import json

import requests
import re


class HttpRequest:
    def __init__(self, string_image):
        self.__image = string_image
        self.__image = self.__image.decode('utf-8')
        self.__apiKey = None
        self.__dataFromImage = None
        self.__dataNLP = None
        self.__dataOCR = {
              "requests": [
                {
                  "features": [
                    {
                      "maxResults": 50,
                      "model": "builtin/latest",
                      "type": "DOCUMENT_TEXT_DETECTION"
                    }
                  ],
                  "image": {
                    "content": self.__image
                  }
                }
              ]
            }



    def __get_api_key(self):
        with open('apikey.txt') as f:
            self.__apiKey = f.readline()

    def __get_response_from_google_ocr(self):
        url = "https://vision.googleapis.com/v1/images:annotate" + "?key=" + self.__apiKey

        response = requests.post(url, json.dumps(self.__dataOCR))
        json_response = response.json()
        description = json_response.get("responses")[0].get("textAnnotations")[0].get("description")
        self.__dataFromImage = description

    def get_response_from_google_nlp(self):
        self.__get_api_key()

        self.__get_response_from_google_ocr()

        match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', self.__dataFromImage)

        self.__dataNLP = {
            "document": {
                "content": self.__dataFromImage,
                "type": "PLAIN_TEXT"
            },
            "features": {
                "extractSyntax": False,
                "extractEntities": True,
                "extractDocumentSentiment": False,
                "extractEntitySentiment": False,
                "classifyText": False
            }
        }

        url = "https://language.googleapis.com/v1/documents:annotateText" + "?key=" + self.__apiKey

        response = requests.post(url, json.dumps(self.__dataNLP))
        response_dict = {}
        json_response = response.json()
        entities = json_response.get("entities")
        for el in entities:
            # print(el)
            if el.get("type") not in response_dict.keys():
                response_dict[el.get("type")] = el.get("name")

        response_dict["EMAIL"] = match.group(0)
        # print(response_dict)
        dict_to_json = json.dumps(response_dict)
        return dict_to_json

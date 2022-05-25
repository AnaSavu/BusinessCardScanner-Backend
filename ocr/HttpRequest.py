import json
from types import SimpleNamespace

import requests


class HttpRequest:
    def __init__(self, string_image):
        self.__image = string_image
        self.__image = self.__image.decode('utf-8')
        self.__apiKey = None
        self.__data = {
              "requests": [
                {
                  "features": [
                    {
                      "maxResults": 50,
                      "type": "OBJECT_LOCALIZATION"
                    },
                    {
                      "maxResults": 50,
                      "type": "LABEL_DETECTION"
                    },
                    {
                      "maxResults": 50,
                      "model": "builtin/latest",
                      "type": "DOCUMENT_TEXT_DETECTION"
                    },
                    {
                      "maxResults": 50,
                      "type": "SAFE_SEARCH_DETECTION"
                    }
                  ],
                  "image": {
                    "content": self.__image
                  },
                  "imageContext": {
                    "cropHintsParams": {
                      "aspectRatios": [
                        0.8,
                        1,
                        1.2
                      ]
                    }
                  }
                }
              ]
            }



    def __getApiKey(self):
        with open('apikey.txt') as f:
            self.__apiKey = f.readline()

    def getHttpResponse(self):
        self.__getApiKey()

        url = "https://vision.googleapis.com/v1/images:annotate"  + "?key=" + self.__apiKey

        response = requests.post(url, json.dumps(self.__data))
        # x = json.loads(str(response), object_hook=lambda d: SimpleNamespace(**d))
        json_response = response.json()
        print(json_response)
        description = json_response.get("responses")[0].get("textAnnotations")[0].get("description")
        return description
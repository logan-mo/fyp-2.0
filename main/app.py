import requests
import cv2


# def send_request_to_container(file, file_name):

#     url = f"https://2f4d-35-232-64-33.ngrok-free.app/process"

#     payload = {}
#     files = [("file", (file_name, file, "image/png"))]
#     headers = {}

#     response = requests.request(
#         "POST", url, headers=headers, data=payload, files=files, timeout=30
#     )

#     return response


# from glob import glob

# for file_path in glob("val/*.png"):

#     image = cv2.imread(file_path)
#     file = cv2.imencode(".png", image)[1].tobytes()

#     response = send_request_to_container(file, file_path)
#     print(response)
#     print(response.text)

# Run the above code, but in parallel


from concurrent.futures import ThreadPoolExecutor
from glob import glob
from datetime import datetime


def send_request_to_container(file, file_name):

    url = f"https://2f4d-35-232-64-33.ngrok-free.app/process"

    payload = {}
    files = [("file", (file_name, file, "image/png"))]
    headers = {}

    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files, timeout=30
    )

    return response


def process_image(file_path):
    image = cv2.imread(file_path)
    file = cv2.imencode(".png", image)[1].tobytes()
    response = send_request_to_container(file, file_path)
    print(response)
    print(response.text)


start = datetime.now()

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_image, glob("val/*.png"))

print("Total Time: ", datetime.now() - start)

from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form

from ultralytics import YOLO
from PIL import Image
import ultralytics
import easyocr
import torch

from typing import List, Tuple, Union, Dict
import numpy as np
import datetime
import psycopg2
import random
import uuid
import cv2
import os

from copy import deepcopy

app = FastAPI()

## app.include_router(get_info.router)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_IMG_DIM = (640, 640)
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
DEVICE = torch.device(DEVICE)

print("Loading models...")

MODEL_BASE_DIR = "saved_models"
CAR_LABEL_DETECTOR_PATH = os.path.join(
    MODEL_BASE_DIR, "detector_pytorch", "car_detection_yolov9e.pt"
)
CAR_COLOR_CLASSIFIER_PATH_PATH = os.path.join(
    MODEL_BASE_DIR, "detector_pytorch", "color_classification_yolov8x.pt"
)
car_label_detector = YOLO(CAR_LABEL_DETECTOR_PATH)
print("Car label detector loaded.")
car_color_classifier = YOLO(CAR_COLOR_CLASSIFIER_PATH_PATH)
print("Car color classifier loaded.")
number_plate_reader_ocr = easyocr.Reader(["en"])
print("Number plate reader loaded.")
print("Models loaded.")

CAR_LABELS = [
    "bus",
    "car_honda",
    "car_hyundai",
    "car_other",
    "car_suzuki",
    "car_toyota",
    "number_plate",
    "van_honda",
    "van_hyundai",
    "van_other",
    "van_suzuki",
    "van_toyota",
]
NUMBER_PLATE_LABEL = "number_plate"
COLOR_NAMES = ["black", "blue", "green", "grey", "red", "silver", "white"]

PLOT = True
COLORS = np.random.randint(0, 255, size=(len(CAR_LABELS), 3))

# DB_CONNECTION_PARAMS = {
#     "database": "postgres",
#     "user": "postgres",
#     "password": "root",
#     "host": "localhost",
#     "port": "5432",
# }


def crop_image_lrtb(image: np.ndarray, crop_region_lrtb: List[int]):
    x1, y1, x2, y2 = crop_region_lrtb

    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def detect_car_labels(image: np.ndarray) -> List:
    image = cv2.resize(image, YOLO_IMG_DIM, interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(image).to(DEVICE)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, axis=0)
    image = image / 255.0
    results = car_label_detector(image)
    return results


def save_image_to_path(image: np.ndarray, path: str) -> None:
    image = Image.fromarray(image)
    image.save(path)


def detect_relevant_objexts(
    frame: np.ndarray,
) -> List[ultralytics.engine.results.Results]:
    results = detect_car_labels(frame)
    return results


def classify_car_color(image: np.ndarray) -> List:
    image = cv2.resize(image, YOLO_IMG_DIM, interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(image).to(DEVICE)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, axis=0)
    image = image / 255.0
    results = car_color_classifier(image)
    return results


def read_number_plate(
    frame: np.ndarray, xyxy: Union[List[Union[int, float]], torch.FloatTensor]
) -> Dict:
    x1, y1, x2, y2 = map(int, xyxy)

    x1, y1, x2, y2 = (
        int(x1 * frame.shape[1] / YOLO_IMG_DIM[0]),
        int(y1 * frame.shape[0] / YOLO_IMG_DIM[1]),
        int(x2 * frame.shape[1] / YOLO_IMG_DIM[0]),
        int(y2 * frame.shape[0] / YOLO_IMG_DIM[1]),
    )

    number_plate_img = frame[y1:y2, x1:x2]

    result = number_plate_reader_ocr.readtext(number_plate_img)

    if len(result) != 0:
        bbox, text, prob = result[0]
        return {"text": text, "prob": prob}
    return {"text": None, "prob": None}


def intersects_lrtb(box1, box2):
    """
    box:
        (xmin, ymin, xmax, ymax)
    """
    x1, y1, x2, y2 = box1  # face
    x3, y3, x4, y4 = box2  # person

    if x1 > x3 and y1 > y3 and x2 < x4 and y2 < y4:
        return True
    return False


@app.get("/")
def health_check():
    return {"message": "working"}


@app.post("/process")
def process(input_frame: UploadFile = File(...)):
    print("Recieved Image.")
    image = np.fromstring(input_frame.file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = detect_relevant_objexts(img)

    cars = []
    number_plates = []

    for result in results:
        cls = result.boxes.cls
        conf = result.boxes.conf
        xyxy = result.boxes.xyxy

        for idx, bbox in enumerate(xyxy):
            if cls[idx] == CAR_LABELS.index(NUMBER_PLATE_LABEL):
                number_plates.append(idx)
            else:
                cars.append(idx)

    response = []
    for car_idx in cars:
        car_bbox = results[0].boxes.xyxy[car_idx]
        car_cls = results[0].boxes.cls[car_idx]
        car_conf = results[0].boxes.conf[car_idx]

        registration_number = None
        registration_number_conf = None

        car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)
        res = classify_car_color(crop_image_lrtb(img, [car_x1, car_y1, car_x2, car_y2]))
        car_color = res[0].names[res[0].probs.top1]

        for number_plate_idx in number_plates:
            number_plate_bbox = results[0].boxes.xyxy[number_plate_idx]
            number_plate_cls = results[0].boxes.cls[number_plate_idx]
            number_plate_conf = results[0].boxes.conf[number_plate_idx]

            if intersects_lrtb(number_plate_bbox, car_bbox):
                read_results = read_number_plate(
                    img, results[0].boxes.xyxy[number_plate_idx]
                )
                registration_number = read_results["text"]
                registration_number_conf = read_results["prob"]

        x1, y1, x2, y2 = map(int, car_bbox.cpu().numpy().tolist())
        x1, y1, x2, y2 = (
            int(x1 * img.shape[1] / YOLO_IMG_DIM[0]),
            int(y1 * img.shape[0] / YOLO_IMG_DIM[1]),
            int(x2 * img.shape[1] / YOLO_IMG_DIM[0]),
            int(y2 * img.shape[0] / YOLO_IMG_DIM[1]),
        )

        response.append(
            {
                "car_bbox": [x1, y1, x2, y2],
                "car_cls": car_cls.cpu().item(),
                "car_conf": car_conf.cpu().item(),
                "car_color": car_color,
                "registration_number": registration_number,
                "registration_number_conf": (
                    registration_number_conf.cpu().item()
                    if registration_number_conf is not None
                    else None
                ),
            }
        )

    return {"response": response}


if __name__ == "__main__":

    video_file = input("Enter video file path: ")

    print(f"Processing video: {video_file}")

    cap = cv2.VideoCapture(video_file)
    video_writer = cv2.VideoWriter(
        ("processed_" + video_file).replace("mp4", "avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        20,
        (int(cap.get(3)), int(cap.get(4))),
    )

    frame_count = 0
    try:

        while cap.isOpened():
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}")

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 20 != 0:
                continue

            results = detect_relevant_objexts(frame)

            cars = []
            number_plates = []

            for result in results:
                cls = result.boxes.cls
                conf = result.boxes.conf
                xyxy = result.boxes.xyxy

                for idx, bbox in enumerate(xyxy):
                    if cls[idx] == CAR_LABELS.index(NUMBER_PLATE_LABEL):
                        number_plates.append(idx)
                    else:
                        cars.append(idx)

            response = []
            for car_idx in cars:
                car_bbox = results[0].boxes.xyxy[car_idx]
                car_cls = results[0].boxes.cls[car_idx]
                car_conf = results[0].boxes.conf[car_idx]

                registration_number = None
                registration_number_conf = None

                car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)
                res = classify_car_color(
                    crop_image_lrtb(frame, [car_x1, car_y1, car_x2, car_y2])
                )
                car_color = res[0].names[res[0].probs.top1]

                for number_plate_idx in number_plates:
                    number_plate_bbox = results[0].boxes.xyxy[number_plate_idx]
                    number_plate_cls = results[0].boxes.cls[number_plate_idx]
                    number_plate_conf = results[0].boxes.conf[number_plate_idx]

                    if intersects_lrtb(number_plate_bbox, car_bbox):
                        read_results = read_number_plate(
                            frame, results[0].boxes.xyxy[number_plate_idx]
                        )
                        registration_number = read_results["text"]
                        registration_number_conf = read_results["prob"]

                response.append(
                    {
                        "car_bbox": car_bbox,
                        "car_cls": car_cls,
                        "car_conf": car_conf,
                        "car_color": car_color,
                        "registration_number": registration_number,
                        "registration_number_conf": registration_number_conf,
                    }
                )

            for x in response:
                print(x)

            for result in results:
                for idx, bbox in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1, x2, y2 = (
                        int(x1 * frame.shape[1] / YOLO_IMG_DIM[0]),
                        int(y1 * frame.shape[0] / YOLO_IMG_DIM[1]),
                        int(x2 * frame.shape[1] / YOLO_IMG_DIM[0]),
                        int(y2 * frame.shape[0] / YOLO_IMG_DIM[1]),
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"{CAR_LABELS[int(result.boxes.cls[idx].cpu().item())]}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

            video_writer.write(frame)

    except Exception as e:
        ...
    finally:
        cap.release()
        video_writer.release()

        print("Done processing videos.")

    # img = cv2.imread("c95a071a-best_pics_99.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # results = detect_relevant_objexts(img)

    # cars = []
    # number_plates = []

    # for result in results:
    #     cls = result.boxes.cls
    #     conf = result.boxes.conf
    #     xyxy = result.boxes.xyxy

    #     for idx, bbox in enumerate(xyxy):
    #         if cls[idx] == CAR_LABELS.index(NUMBER_PLATE_LABEL):
    #             number_plates.append(idx)
    #         else:
    #             cars.append(idx)

    # response = []
    # for car_idx in cars:
    #     car_bbox = results[0].boxes.xyxy[car_idx]
    #     car_cls = results[0].boxes.cls[car_idx]
    #     car_conf = results[0].boxes.conf[car_idx]

    #     registration_number = None
    #     registration_number_conf = None

    #     car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)
    #     res = classify_car_color(crop_image_lrtb(img, [car_x1, car_y1, car_x2, car_y2]))
    #     car_color = res[0].names[res[0].probs.top1]

    #     for number_plate_idx in number_plates:
    #         number_plate_bbox = results[0].boxes.xyxy[number_plate_idx]
    #         number_plate_cls = results[0].boxes.cls[number_plate_idx]
    #         number_plate_conf = results[0].boxes.conf[number_plate_idx]

    #         if intersects_lrtb(number_plate_bbox, car_bbox):
    #             read_results = read_number_plate(
    #                 img, results[0].boxes.xyxy[number_plate_idx]
    #             )
    #             registration_number = read_results["text"]
    #             registration_number_conf = read_results["prob"]

    #     response.append(
    #         {
    #             "car_bbox": car_bbox,
    #             "car_cls": car_cls,
    #             "car_conf": car_conf,
    #             "car_color": car_color,
    #             "registration_number": registration_number,
    #             "registration_number_conf": registration_number_conf,
    #         }
    #     )

    # for x in response:
    #     print(x)

    # for result in results:
    #     for idx, bbox in enumerate(result.boxes.xyxy):
    #         x1, y1, x2, y2 = map(int, bbox)
    #         x1, y1, x2, y2 = (
    #             int(x1 * img.shape[1] / YOLO_IMG_DIM[0]),
    #             int(y1 * img.shape[0] / YOLO_IMG_DIM[1]),
    #             int(x2 * img.shape[1] / YOLO_IMG_DIM[0]),
    #             int(y2 * img.shape[0] / YOLO_IMG_DIM[1]),
    #         )

    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
    #         cv2.putText(
    #             img,
    #             f"{CAR_LABELS[int(result.boxes.cls[idx].cpu().item())]}",
    #             (x1, y1 - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0, 0, 0),
    #             1,
    #         )

    # cv2.imwrite("output.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

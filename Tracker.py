import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision import Color, Detections
import cv2

HOME = os.getcwd()

# The name for the project
PROJECT_NAME = f"Tracker"

# The path for the project folder
PROJECT_DIR = Path(f"./{PROJECT_NAME}")

# Create the project directory if it does not already exist
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# Default GOOGLE DRIVE
GDRIVE_VID_PATH = f"/content/drive/MyDrive/detect_human/Samples"

# gdown "https://drive.google.com/uc?id=1D8HO3ina1HXOIrZpyJ4gpRgBPZlssmmK&confirm=t"
# gdown "https://drive.google.com/uc?id=1yqUPkFXFF_5AOi6QaMxyonGazA-Fe9tN&confirm=t"
# gdown "https://drive.google.com/uc?id=1LZ_uasbiu7nZbo8Mef5zAxJjfxyWrCZW&confirm=t"
# gdown "https://drive.google.com/uc?id=1aMh7HGmWk4OriycZqd_E8aizKh9Xlht_&confirm=t"
# gdown "https://drive.google.com/uc?id=13yXJxe9L0r3_i5cLaM7MhPwf3Vl3-vq_&confirm=t"
# gdown "https://drive.google.com/uc?id=1Lrgjr_womyOe_cEDkU-Jx4eLSmQUd0_v&confirm=t"
# gdown "https://drive.google.com/uc?id=18F0YAJj3AY65OfF-hUgr56Tzg-z0431x&confirm=t"

SOURCE_VIDEO_PATH = f"{HOME}/{PROJECT_NAME}/test.mp4"
TARGET_VIDEO_PATH = f"{HOME}/{PROJECT_NAME}/result.mp4"

MODEL_WEIGHT = 'best_yolov8m_59000.pt'

faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['M', 'F']


def predict_age_gender(frame, detections: Detections, padding: int):
    ages = np.array([])
    genders = np.array([])

    boxes = detections.xyxy

    for box in boxes:
        feature = frame[max(0, int(box[1]) - padding): min(int(box[3]) + padding, frame.shape[0]),
                  max(0, int(box[0]) - padding): min(int(box[2]) + padding, frame.shape[1])]
        blob = cv2.dnn.blobFromImage(feature, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        gender_predictions = genderNet.forward()
        gender = genderList[gender_predictions[0].argmax()]

        ageNet.setInput(blob)
        age_predictions = ageNet.forward()
        age_cls = age_predictions[0].argmax()

        if age_cls < 3:
            age = 'k'
        elif 3 < age_cls < 6:
            age = 'y'
        else:
            age = 'o'

        genders = np.append(genders, gender)
        ages = np.append(ages, age)

    return ages, genders


model = YOLO(MODEL_WEIGHT)
model.fuse()

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

byte_tracker = sv.ByteTrack(
    track_thresh=0.1,
    track_buffer=900,
    match_thresh=0.8
)

# botsort_tracker: BOTSORT = ultralytics.trackers.BOTSORT(byte_tracker)

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5,
    text_color=Color.green()
)


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    padding = 20

    ages, genders = predict_age_gender(frame, detections, padding)
    labels = [
        f"{tracker_id} / {age} / {gender}"
        for tracker_id, age, gender
        in zip(detections.tracker_id, ages, genders)
    ]

    return box_annotator.annotate(frame.copy(), detections=detections, labels=labels)


if __name__ == '_main__':
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )

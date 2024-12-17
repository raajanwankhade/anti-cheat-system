
import cv2
import mediapipe as mp
import numpy as np
from FaceDetailsCalculator import FaceDetails

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the face landmark model
model_path = 'face_landmarker.task'  # Update this to your actual model path

def print_landmark_details(result: FaceLandmarkerResult, input_image: mp.Image, timestamp_ms: int):
    face_details = FaceDetails(result, input_image.numpy_view())
    print(f"\rFACE DETAILS:\nIris Position: {face_details.iris_pos},\nIris Ratio: {face_details.iris_ratio},\nMouth Area: {face_details.mouth_area},\nMouth Zone: {face_details.mouth_zone},\nRadial Distance: {face_details.radial_distance},\nGaze Direction: {face_details.gaze_direction},\nGaze Zone: {face_details.gaze_zone}")

# Configure FaceLandmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_landmark_details)

# Open webcam
cap = cv2.VideoCapture(0)

# Create FaceLandmarker
with FaceLandmarker.create_from_options(options) as landmarker:
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert OpenCV frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Process frame
        landmarker.detect_async(mp_image, frame_count)
        frame_count += 1

        # Display frame (optional)
        cv2.imshow('Face Landmark Detection', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
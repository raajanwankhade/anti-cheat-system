
import cv2
import mediapipe as mp
import numpy as np
from FaceDetailsCalculator import FaceDetails
from deepface import DeepFace
import threading
from tensorflow.keras import backend as K

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

src_image = r"C:\Users\singl\Desktop\Bhuvanesh\NITK\SEM4\IT255_AI\Project Files\Bhuvanesh_img.jpg"
# Path to the face landmark model
model_path = 'face_landmarker.task'  # Update this to your actual model path


def async_verify(frame):
    try:
        frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        referenceframe = cv2.imread(r"VerifiedLiveFrame.jpg")
        referenceframe = cv2.cvtColor(referenceframe, cv2.COLOR_BGR2RGB)
        result = DeepFace.verify(img1_path=frame_new, img2_path=referenceframe, detector_backend='mediapipe', model_name='ArcFace')
        print("Verification Result:", result['verified'])
    except Exception as e:
        print("Error in DeepFace verification:", e)
    finally:
        K.clear_session()

def print_landmark_details(result: FaceLandmarkerResult, input_image: mp.Image, timestamp_ms: int):
    face_details = FaceDetails(result, input_image.numpy_view())
    print(f"Iris Position: {face_details.iris_pos},Mouth Zone: {face_details.mouth_zone},Gaze Direction: {face_details.gaze_direction},Gaze Zone: {face_details.gaze_zone}")


# Configure FaceLandmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_landmark_details)

verification_interval = 3  # seconds
last_verification_time = 0

# getGoodCapture(src_image)

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
        
        frame_count += 1

        # Display frame (optional)
        cv2.imshow('Face Landmark Detection', frame)

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - last_verification_time >= verification_interval:
            last_verification_time = current_time
            threading.Thread(target=async_verify, args=(frame.copy(),)).start()
            landmarker.detect_async(mp_image, frame_count)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
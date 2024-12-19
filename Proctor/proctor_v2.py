
import cv2
from deepface import DeepFace
import numpy as np
# from facial_landmarking_utils import face_pose
from handpose import inference
import mediapipe as mp
from cheat_prob import calculate_cheat_score
from FaceDetailsCalculator import FaceDetails
import torch 

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the face landmark model
model_path = 'face_landmarker.task'  # Update this to your model path

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def enhance_frame(image):
    alpha = 1.3
    beta = 30
    image = cv2.convertScaleAbs(image, alpha = alpha, beta=beta)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    new = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return new



class LiveProctor:
    def __init__(self, target_path, yolo_model, media_pipe):
        self.target_path = target_path
        self.yolo_model = yolo_model
        self.media_pipe = media_pipe
        self.output = {}
        self.frame_count = 0
        
        # Initialize MediaPipe
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.landmark_callback)
            
        self.landmarker = FaceLandmarker.create_from_options(self.options)

    def landmark_callback(self, result: FaceLandmarkerResult, input_image: mp.Image, timestamp_ms: int):
        if result and result.face_landmarks:
            face_details = FaceDetails(result, input_image.numpy_view())
            self.output['Face Direction'] = face_details.gaze_direction
            self.output['Face Zone'] = face_details.gaze_zone
            self.output['Eye Direction'] = face_details.iris_pos
            self.output['Mouth'] = face_details.mouth_zone
            self.output['Number of people'] = face_details.num_faces

    def process_frame(self, frame):
        # Hand detection
        hand_dict = inference(frame, self.yolo_model, self.media_pipe)
        if hand_dict:
            # Update with safe key access
            hand_keys = {
                'hand_detected': 'Hand Detected',
                'prohibited_item_use': 'Prohibited Item Use', 
                'distance': 'Distance',
                'illegal_objects': 'Illegal Objects',
                'prohibited_item': 'Prohibited Item'
            }
            
            for src_key, dst_key in hand_keys.items():
                if src_key in hand_dict:
                    self.output[dst_key] = hand_dict[src_key]

        # Face verification
        processed_frame = enhance_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        target = cv2.imread(self.target_path)
        self.output['Identity'] = DeepFace.verify(
            img1_path=target, 
            img2_path=processed_frame,
            model_name='ArcFace',
            detector_backend='mediapipe',
            normalization='ArcFace',
            align=False,
            enforce_detection=False)['verified']

        # Face landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, self.frame_count)
        self.frame_count += 1

        # Calculate cheat score
        self.output['Cheat Score'] = calculate_cheat_score(self.output)
        return self.output

    def start_stream(self):
        cap = cv2.VideoCapture(0)
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                result = self.process_frame(frame)
                
                # Display results
                cv2.imshow('Live Proctoring', frame)
                print(f"\rCurrent state: {result}", end='')
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                          path='/home/shanveen-ortho-clinic/Documents/Projects/Proctor/anti-cheat-system/YOLO_weights/best.pt', #change this acc to ur path
                          force_reload=False).to(device)
    mpHands = mp.solutions.hands
    media_pipe_dict = {
        'mpHands': mpHands,
        'hands': mpHands.Hands(static_image_mode=False, 
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5),
        'mpdraw': mp.solutions.drawing_utils
    }

    target_path = '/home/shanveen-ortho-clinic/Documents/Projects/Proctor/anti-cheat-system/Proctor/Images/identity.jpeg'
    proctor = LiveProctor(target_path, model, media_pipe_dict)
    proctor.start_stream()
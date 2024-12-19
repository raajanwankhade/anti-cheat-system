import cv2
from deepface import DeepFace
import numpy as np
from facial_landmarking_utils import face_pose
from handpose import inference
import torch
import mediapipe as mp
from cheat_prob import calculate_cheat_score

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def numoffaces(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    return len(faces)

def enhance_frame(image):
    alpha = 1.3
    beta = 30
    image = cv2.convertScaleAbs(image, alpha = alpha, beta=beta)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    new = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return new

def face_module(frame_path, target_path, yolo_model, media_pipe):
    '''
    Takes location as input of current frame from camera and the location of target image, and returns the output
    '''
    frame = cv2.imread(frame_path)
    hand_dict = inference(frame, yolo_model, media_pipe)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    frame = enhance_frame(frame)

    target = cv2.imread(target_path)
    #target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    outcome = DeepFace.verify(img1_path=target, img2_path=frame, model_name='ArcFace', detector_backend='mediapipe', normalization='ArcFace', align=False, enforce_detection = False)['verified']
    num_faces = numoffaces(frame)
    face_dict = face_pose(frame)
    #print(hand_dict)
    output = {}
    output['Identity'] = outcome
    output['Number of people'] = num_faces
    if hand_dict:
        output['Hand Detected'] = hand_dict['hand_detected']
        output['Prohibited Item Use'] = hand_dict['Prohibited Item Use']
        output['Distance'] = hand_dict['distance']
        output['Illegal Objects'] = hand_dict['illegal_objects']
        output['Prohibited Item'] = hand_dict['prohibited_item']
    if face_dict:
        output['Face Direction'] = face_dict['dir_text']
        output['Face Zone'] = face_dict['colour_zone']
        output['Eye Direction'] = face_dict['iris_pos']
        output['Mouth'] = face_dict['mouth_zone']

    output['Cheat Score'] = calculate_cheat_score(output)
    return output

def start(frame_path, target_path):
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    frame = enhance_frame(frame)

    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    outcome = DeepFace.verify(img1_path=target, img2_path=frame, model_name='ArcFace', detector_backend='mediapipe', normalization='ArcFace', align=False,  enforce_detection = False)['verified']

    if outcome:
        cv2.imwrite(target_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) ## this will be new target.png which will be tested

    return outcome

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/shanveen-ortho-clinic/Documents/Projects/Proctor/anti-cheat-system/YOLO_weights/best.pt', force_reload=False).to(device)
    mpHands = mp.solutions.hands
    media_pipe_dict = {
        'mpHands': mpHands,
        'hands': mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5),
        'mpdraw': mp.solutions.drawing_utils
    }

    frame_path = '/home/shanveen-ortho-clinic/Documents/Projects/Proctor/anti-cheat-system/Proctor/Images/verify2.jpeg'
    target_path = '/home/shanveen-ortho-clinic/Documents/Projects/Proctor/anti-cheat-system/Proctor/Images/identity.jpeg'
    result = face_module(frame_path, target_path, yolo_model=model, media_pipe=media_pipe_dict)
    print(result)

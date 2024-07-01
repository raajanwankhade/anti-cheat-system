import cv2
from deepface import DeepFace
import numpy as np
from .facial_landmarking_utils import face_pose

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
test_img = ''
def numoffaces(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
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

def face_module(frame, target_path):
    '''
    Takes location as input of current frame from camera and the location of target image, and returns the output
    '''
    start = time.time()
    # frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    frame = enhance_frame(frame)

    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    outcome = DeepFace.verify(img1_path=target, img2_path=frame, model_name='ArcFace', detector_backend='mediapipe', normalization='ArcFace', align=False)['verified']
    num_faces = numoffaces(frame)
    face_dict = face_pose(frame)

    end = time.time()

    output = {}
    output['Identity'] = outcome
    output['Number of people'] = num_faces
    try:
        output['Face Direction'] = face_dict['dir_text']
        output['Face Zone'] = face_dict['colour_zone']
        output['Eye Direction'] = face_dict['iris_pos']
        output['Mouth'] = face_dict['mouth_zone']
    except:
        output['Identity'] = False
        output['Face Direction'] = None
        output['Face Zone'] = None
        output['Eye Direction'] = None
        output['Mouth'] = None
    output['Time'] = np.round(end - start, 2)
    return output

def start(frame, target_path):
    # frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    frame = enhance_frame(frame)

    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    outcome = DeepFace.verify(img1_path=target, img2_path=frame, model_name='ArcFace', detector_backend='mediapipe', normalization='ArcFace', align=False)['verified']
    
    if outcome:
        cv2.imwrite(target_path, frame) ## this will be new target.png which will be tested
    
    return outcome

def face_wrapper(target_path):
    global test_img
    while True:
        output = face_module(test_img, target_path)
        print(output, end = '\r')
        time.sleep(1)



if __name__ == '__main__':
    import threading
    import time
    target_path = input("Enter target path: ")
    outcome = False
    while not outcome:
        print("Searching for examinee...")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow("Initial Capture, press [q]", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows() 
        outcome = start(frame, target_path)
    
    print("Examinee verified")
    test_img = frame

    face_thread = threading.Thread(target=face_wrapper, args=(target_path,))
    face_thread.daemon = True
    face_thread.start()

    cap = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cap.read()
        test_img = frame.copy()
        cv2.imshow("Proctor", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except:
        continue

print("\nthanks for attending")

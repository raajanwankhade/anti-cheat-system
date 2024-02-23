import threading
import time
import cv2
from deepface import DeepFace
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def numoffaces(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return len(faces)

def enhance_frame(frame):
    alpha = 1.3
    beta = 30
    frame = cv2.convertScaleAbs(frame, alpha = alpha, beta=beta)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    new = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    return new

def verify():
    global name, frame
    try:

        outcome = DeepFace.verify(img1_path=target, img2_path=frame, model_name='ArcFace', detector_backend='mediapipe', normalization='ArcFace', align=False)

        print(outcome)
        if outcome['verified']:
            name = examinee
        else:
            name = 'PROXY'
        print(name)
    except Exception as e:
        print(e)
        pass
    

print("INITIALISING PROCTOR...")
frame = None #global var
examinee = input("Enter your name here: ")
name = '' #global var
msg = ''
target = input("please put path to registered image: ")


# to initialize the frame
while name != examinee:
    print("Searching for examinee...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = enhance_frame(frame)
        cv2.imshow("Initial Capture, press [q]", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 
    verify()

print("Examinee confirmed, starting exam...")
target = frame


def face_recog():
    global frame, msg
    while True:
        num = numoffaces(frame)
        if num>1:
            msg = "Multiple faces in hall"
        else:
            msg = "1 face"
        verify()
        # print(name)
        time.sleep(1)

recog_thread = threading.Thread(target = face_recog)
recog_thread.daemon = True
recog_thread.start()

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = enhance_frame(frame)
        cv2.putText(frame, name,(100,50),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
        cv2.putText(frame, msg,(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
        cv2.imshow("Proctor", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except:
        continue
cap.release()
cv2.destroyAllWindows()   

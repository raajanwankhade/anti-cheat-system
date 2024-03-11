import threading
import time
import cv2
from deepface import DeepFace
import numpy as np
from facial_landmarking_utils import face_pose

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

def verify(image):
    global name, target
    try:
        outcome = DeepFace.verify(img1_path=target, img2_path=image, model_name='ArcFace', detector_backend='mediapipe', normalization='ArcFace', align=False)

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
msg = '' #global
dir_text = ''
radius = ''
colour_zone = ''
iris_pos = ''
eye_ratio = ''
mouth_area = ''
mouth_zone = ''

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
    verify(frame)

print("Examinee confirmed, starting exam...")
target = frame.copy()


def face_recog():
    global msg, frame
    while True:
        num = numoffaces(frame)
        if num>1:
            msg = "Multiple faces in hall"
        else:
            msg = "1 face"
        verify(frame)
        time.sleep(2)

def getPose():
    global frame, dir_text, radius, colour_zone, iris_pos, eye_ratio, mouth_area, mouth_zone
    while True:
        face_dict = face_pose(frame)

        if face_dict:
            success = face_dict['success']
            if success:
                dir_text = face_dict['dir_text']
                radius = face_dict['radius']
                colour_zone = face_dict['colour_zone']
                iris_pos = face_dict['iris_pos']
                eye_ratio = face_dict['ratio']
                mouth_area = face_dict['mouth_area']
                mouth_zone = face_dict['mouth_zone']
# def face_recog(frame):
#     global msg
#     print(f"Face recog called")
#     num = numoffaces(frame)
#     if num>1:
#         msg = "Multiple faces in hall"
#     else:
#         msg = "1 face"
#     verify(frame)
#     print("verified")
#     showDetails(frame)
#     # print(name)
#     # time.sleep(5)

# def getPose(frame):
#     global dir_text, radius, colour_zone, iris_pos, eye_ratio
#     print(f"Get Pose called")
#     face_dict = face_pose(frame)

#     if face_dict:
#         success = face_dict['success']
#         if success:
#             print("Success")
#             dir_text = face_dict['dir_text']
#             radius = face_dict['radius']
#             colour_zone = face_dict['colour_zone']
#             iris_pos = face_dict['iris_pos']
#             eye_ratio = face_dict['ratio']

#     showDetails(frame)

def showDetails():
    
    global frame, dir_text, radius, colour_zone, iris_pos, eye_ratio, name, msg, mouth_area, mouth_zone
    try:
        cv2.putText(frame, name,(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
        cv2.putText(frame, msg,(10,75),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
        cv2.putText(frame, dir_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "radius: " + str(np.round(radius,2)), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "zone: " + colour_zone, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "eye_position: " + iris_pos, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "ratio: " + str(np.round(eye_ratio,2)), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "mouth: " + str(np.round(mouth_area,2)), (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "mouth zone: " + mouth_zone, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    except:
        pass


recog_thread = threading.Thread(target = face_recog)
land_thread = threading.Thread(target=getPose)
land_thread.daemon = True
recog_thread.daemon = True
recog_thread.start()
land_thread.start()
cap = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = enhance_frame(frame)
        showDetails()
        cv2.imshow("Proctor", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except:
        continue
    

# timer_list = []
# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     frame = enhance_frame(frame)

#     cv2.imshow("Proctor", frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

#     if len(timer_list) == 0:
#         recog_thread = threading.Timer(interval=5,function = face_recog, args = (frame,))
#         land_thread = threading.Timer(interval=1,function=getPose, args = (frame,))

#         timer_list.extend([recog_thread,land_thread])

#         for timer in timer_list:
#             timer.start()
#     timer_list = []

#     # showDetails()

cap.release()
cv2.destroyAllWindows()   

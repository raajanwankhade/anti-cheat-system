import cv2
import mediapipe as mp
import numpy as np
import time
import math

# initialising MediaPipe FaceMesh
mp_face_mesh = None
face_mesh = None

# drawing specifications for landmarks and connections
mp_drawing = None
drawing_spec = None

#IRIS indices
RIGHT_IRIS = None
LEFT_IRIS = None
L_H_LEFT = None
L_H_RIGHT =None
R_H_LEFT = None
R_H_RIGHT = None

def initialize_globals():
    global mp_face_mesh, face_mesh,mp_drawing,drawing_spec,RIGHT_IRIS,LEFT_IRIS, L_H_LEFT, L_H_RIGHT, R_H_LEFT,R_H_RIGHT, INNER_LIPS
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1,min_detection_confidence=0.5, min_tracking_confidence = 0.5,refine_landmarks=True)  ##modified here

    # drawing specifications for landmarks and connections
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    #IRIS indices
    RIGHT_IRIS = [474,475,476,477]
    LEFT_IRIS = [469,470,471,472]
    L_H_LEFT = [33] # right eye right most mark
    L_H_RIGHT = [133] # right eye left most mark
    R_H_LEFT = [362] #left eye right most mark
    R_H_RIGHT = [263] #left eye left most mark

    #Mouth indices
    INNER_LIPS = [13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82]


def initialize_camera():
    """
    Initialize the camera and return the VideoCapture object.
    """
    cap = cv2.VideoCapture(0)
    return cap


def process_frame(cap):
    """
    Process a single frame from the camera and return the image.
    """
    success, image = cap.read()
    return success, image


def extract_face_landmarks(image):
    """
    Extract face landmarks using MediaPipe FaceMesh.
    """
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    return results.multi_face_landmarks


def initialize_camera_matrix(img_w, img_h):
    """
    Initialize the camera matrix.

    Parameters:
    - img_w (int): Image width.
    - img_h (int): Image height.

    Returns:
    - np.array: Camera matrix.
    """
    focal_length = 1 * img_w
    return np.array([[focal_length, 0, img_h / 2],
                     [0, focal_length, img_w / 2],
                     [0, 0, 1]])


def calculate_gaze_radius(x, y, z, threshold_white=10, threshold_yellow=15):
    """
    Calculate the radial distance from the center based on head pose angles.

    Parameters:
    - x (float): The rotation angle around the x-axis in degrees.
    - y (float): The rotation angle around the y-axis in degrees.
    - z (float): The rotation angle around the z-axis in degrees.
    - threshold_white (float): Threshold for the white zone.
    - threshold_yellow (float): Threshold for the yellow zone.

    Returns:
    - radial_distance (float): Radial distance from the center.
    - color_zone (str): Color zone ('white', 'yellow', or 'red').
    """
    # Calculate the radial distance from the center based on the angles
    radial_distance = np.sqrt(x**2 + y**2 + z**2)

    # Determine the color zone based on thresholds
    if radial_distance <= threshold_white:
        color_zone = "white"
    elif radial_distance <= threshold_yellow:
        color_zone = "yellow"
    else:
        color_zone = "red"

    return radial_distance, color_zone


def calculate_zone(face_landmarks, image):
    """
    Calculate head pose information and radial zone based on facial landmarks.

    Parameters:
    - face_landmarks (`list`): List of face landmarks obtained from MediaPipe FaceMesh.
    - image (`numpy.ndarray`): The input image frame.

    Returns:
    - success (`bool`): Flag estimating if the pose estimation was successful. 
    - x (`float`): The rotation angle w.r.t. the x-axis in degrees.
    - y (`float`): The rotation angle w.r.t. the y-axis in degrees.
    - z (`float`): The rotation angle w.r.t. the z-axis in degrees.
    - radius (`float`): Radial distance from the center.
    - text (`str`): Direction in which the head is looking ('left', 'right','up', or 'down').
    - colour_zone (`str`): Colour zone ('white', 'yellow', or 'red').

    Note:
    The function extracts facial landmarks from the input image, calculates the 3D head pose,
    and determines the radial distance from the center based on the head pose angles.
    The resulting information includes the rotation angles in x, y, and z axes, as well as
    the radial distance and color zone indicating the direction in which the person is looking.
    """

    img_h, img_w, _ = image.shape

    face_3d = []
    face_2d = []

    if face_landmarks:
        for face_landmark in face_landmarks:
            nose_2d = None
            nose_3d = None
            for idx, lm in enumerate(face_landmark.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # getting the 2D coordinates
                    face_2d.append([x, y])

                    # getting the 3D coordinates
                    face_3d.append([x, y, lm.z])

            # converting face_2d and face_3d to np array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # the camera matrix
            cam_matrix = initialize_camera_matrix(img_w, img_h)

            # the distortion parameters
            # this initialisation means that we are assuming 0 distortion
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # using solvePnP algorithm to find pose of face
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # get Euler angles using RQ Decomposition of the rotation matrix
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # get the y rotation angle in degrees
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # checking where the user's head is tilting
            if y < -10:
                # text = "right" #becuase the camera frame is mirrored
                text = 'left' 
            # elif y < -15:
            #     text = "very_right"
            elif y > 10:
                # text = "left" #because the camera frame is mirrored
                text = 'right' 
            # elif y > 15:
            #     text = "very_left"
            elif x < -8:
                text = "down"
            elif x > 10:
                text = "up"
            else:
                text = "forward"

            # checking the radial zone in which the person is looking
            radius, colour_zone = calculate_gaze_radius(x, y, z)

            return success, x, y, z, radius, text, colour_zone, nose_2d, nose_3d 
    return False, None, None, None, None, None, None, None, None


def euclidean_distance(point1, point2):
    x1,y1 = point1.ravel()
    x2,y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_distance
    iris_position = ""
    if ratio <=0.42:
        iris_position = "right"
    elif 0.42<ratio<=0.57:
        iris_position = 'center'
    else:
        iris_position = 'left'
    return iris_position, ratio

def getIris(face_landmarks, image):
    img_h, img_w = image.shape[:2]
    mesh_points = np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int)for p in face_landmarks[0].landmark])
    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
    center_left = np.array([l_cx,l_cy], dtype = np.int32)
    center_right = np.array([r_cx, r_cy], dtype = np.int32)
    llm = mesh_points[R_H_RIGHT][0]
    lrm = mesh_points[R_H_LEFT][0]
    iris_pos,ratio = iris_position(center_right, llm, lrm)
    return iris_pos, ratio

def getArea(d):
    points = list(d.values())
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def getMouth(face_landmarks, image):
    img_h, img_w = image.shape[:2]
    for face_landmark in face_landmarks:
        lms = face_landmark.landmark
        d = {}
        for index in INNER_LIPS:
            x = int(lms[index].x * img_w)
            y = int(lms[index].y * img_h)
            d[index] = (x,y)
        area = getArea(d)
        zone = ''
        if 0<=area<=50:
            zone = 'GREEN'
        elif 50<area<=150:
            zone = 'YELLOW'
        elif 150<area<=300:
            zone = 'ORANGE'
        else:
            zone = 'RED'
        return area, zone

def face_pose(frame):
    initialize_globals()
    multi_face_landmarks = extract_face_landmarks(frame)
    face_dict = None
    if multi_face_landmarks:
        success, x, y, z, radius, dir_text, colour_zone, nose_2d, nose_3d = calculate_zone(multi_face_landmarks, frame)
        iris_pos, ratio = getIris(multi_face_landmarks, frame)
        m_area, m_zone = getMouth(multi_face_landmarks, frame)
        face_dict = {'success' : success,
                     'x' : x,
                     'y' : y,
                     'z' : z,
                     'radius' : radius,
                     'dir_text' : dir_text,
                     'colour_zone' : colour_zone,
                     'nose_2d' : nose_2d,
                     'nose_3d' : nose_3d,
                     'iris_pos' : iris_pos,
                     'ratio' : ratio,
                     'mouth_area' : m_area,
                     'mouth_zone' : m_zone}
    return face_dict

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        
        face_dict = face_pose(frame)
    #     print(face_dict)
        if face_dict:
            print("level1")
            if face_dict['success']:
                print("level2")
                dir_text = face_dict['dir_text']
                radius = face_dict['radius']
                colour_zone = face_dict['colour_zone']
                iris_pos = face_dict['iris_pos']
                eye_ratio = face_dict['ratio']
                cv2.putText(frame, dir_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(frame, "radius: " + str(np.round(radius,2)), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "zone: " + colour_zone, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "eye_position: " + iris_pos, (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "ratio: " + str(np.round(eye_ratio,2)), (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('Head Pose Estimation', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

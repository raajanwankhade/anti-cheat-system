import cv2
import mediapipe as mp
import numpy as np


class FaceDetails():
    def __init__(self, result, image):
        self.result = result
        self.num_faces = len(result.face_landmarks)
        self.multi_face_landmarks = result.face_landmarks
        self.face_landmarks = None if not self.multi_face_landmarks else result.face_landmarks[0]
        self.image = image
        self.image_h, self.image_w = image.shape[:2]

        self.RIGHT_IRIS = [474,475,476,477]
        self.LEFT_IRIS = [469,470,471,472]
        self.L_H_LEFT = [33] # right eye right most mark
        self.L_H_RIGHT = [133] # right eye left most mark
        self.R_H_LEFT = [362] #left eye right most mark
        self.R_H_RIGHT = [263] #left eye left most mark
        self.INNER_LIPS = [13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82]

        self.iris_pos = ""
        self.iris_ratio = 0
        self.mouth_zone = ""
        self.mouth_area = 0
        self.x_rotation = 0
        self.y_rotation = 0
        self.z_rotation = 0
        self.radial_distance = 0
        self.gaze_direction = ""
        self.gaze_zone = ""

        if self.num_faces == 1:
            if self.face_landmarks is not None:
                self.iris_pos, self.iris_ratio = self.getIris()
                self.mouth_area, self.mouth_zone = self.getMouth()
                self.x, self.y, self.z, self.radial_distance, self.gaze_direction, self.gaze_zone = self.calculate_zone()

        elif self.num_faces > 1:
            print("Multiple faces detected")
        elif self.num_faces == 0:
            print("No face detected")

    # region Iris Calculation
    def euclidean_distance(self, point1, point2):
        distance = np.linalg.norm(point1 - point2)
        return distance

    def iris_position(self, iris_center, right_point, left_point):
        center_to_right_dist = self.euclidean_distance(iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)
        ratio = center_to_right_dist / total_distance
        iris_position = ""
        if ratio <=0.42:
            iris_position = "right"
        elif 0.42<ratio<=0.57:
            iris_position = 'center'
        else:
            iris_position = 'left'
        return iris_position, ratio
    
    def getIris(self):
        mesh_points = np.array([
        np.multiply([p.x, p.y], [self.image_w, self.image_h]).astype(int) 
        for p in self.face_landmarks
        ])
        l_iris_points = [mesh_points[idx] for idx in self.LEFT_IRIS]
        r_iris_points = [mesh_points[idx] for idx in self.RIGHT_IRIS]

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(np.array(l_iris_points))
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(np.array(r_iris_points))

        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        llm = mesh_points[self.R_H_RIGHT]
        lrm = mesh_points[self.R_H_LEFT]

        iris_pos, ratio = self.iris_position(center_right, llm, lrm)

        return iris_pos, ratio
    # endregion Iris Calculation

    # region Mouth Calculation
    def getArea(self, mesh_points):
        points = np.array([mesh_points[idx] for idx in self.INNER_LIPS])
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def getMouth(self):
        mesh_points = np.array([
            np.multiply([p.x, p.y], [self.image_w, self.image_h]).astype(int) 
            for p in self.face_landmarks
        ])
        area = self.getArea(mesh_points)
        zone = ''
        if 0<=area<=160:
            zone = 'GREEN'
        elif 160<area<=500:
            zone = 'YELLOW'
        elif 500<area<=1000:
            zone = 'ORANGE'
        else:
            zone = 'RED'
        return area, zone
    
    # endregion Mouth Calculation

    # region Gaze
    def initialize_camera_matrix(self, img_w, img_h):
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

    def calculate_gaze_radius(self, x, y, z, threshold_white=10, threshold_yellow=15):
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
    

    def calculate_zone(self):
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

        # img_h, img_w, _ = image.shape
        face_2d = []
        face_3d = []
        nose_2d = None
        nose_3d = None

        mesh_points = np.array([
            np.multiply([p.x, p.y], [self.image_w, self.image_h]).astype(float) 
            for p in self.face_landmarks
        ])


        for idx in [33, 263, 1, 61, 291, 199]:  # Specific landmark indices
            x, y = mesh_points[idx]
            z = self.face_landmarks[idx].z

            face_2d.append([x, y])
            face_3d.append([x, y, z * 3000] if idx == 1 else [x, y, z])

            # Capture nose coordinates specifically
            if idx == 1:
                nose_2d = (x, y)
                nose_3d = (x, y, z * 3000)

        # Convert to numpy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)


        # the camera matrix
        cam_matrix = self.initialize_camera_matrix(self.image_w, self.image_h)

   

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
        radius, colour_zone = self.calculate_gaze_radius(x, y, z)
        

        # return success, x, y, z, radius, text, colour_zone, nose_2d, nose_3d
        return x, y, z, radius, text, colour_zone 
    # endregion Gaze
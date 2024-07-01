import cv2 
import math
import mediapipe as mp
from ultralytics import YOLO
import torch


def initialize():
    global mpHands, hands, mpdraw, device, model, mylmList

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpdraw = mp.solutions.drawing_utils
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5','custom', path='best.pt', force_reload=False).to(device)
    mylmList = []

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


# Main loop to process video frames
def handYOLO(frame):
    output = {}
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB for Mediapipe processing

    detections = model(frame)
    yolo_bboxes = []
        
    for detection in detections.pred[0]:
        x1, y1, x2, y2, confidence, class_index = detection.tolist()
        #if model.names[int(class_index)] in ['book', 'cell phone'   ]:
        print(f"Confidence: {confidence} Class: {model.names[int(class_index)]}")
        output['Prohibited Item'] = model.names[int(class_index)]
        yolo_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    results = hands.process(img)  # Process the frame with Mediapipe Hands
    allHands = []
    h, w, c = frame.shape  # Get the height, width, and number of channels of the frame
    
    # Process each detected hand in the frame
    if results.multi_hand_landmarks:
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            myHand = {}
            mylmList = []
            xList = []
            yList = []
            
            # Extract landmark points and store them in lists
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([id, px, py])
                xList.append(px)
                yList.append(py)
            
            # Calculate bounding box around the hand landmarks
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            
            # Store hand information in a dictionary
            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)
            myHand["type"] = handType.classification[0].label
            #if you dont flip the image
            ''' if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
            else:
                        myHand["type"] = "Right"'''
            allHands.append(myHand)
            
            # Draw landmarks and bounding box on the frame
            mpdraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)
            cv2.putText(frame, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    
        for hand in allHands:
            output['hand_detected'] = True
            for yolo_bbox in yolo_bboxes:
                
                x1, y1, x2, y2 = yolo_bbox
                yolo_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                hand_center = hand["center"]
                distance = calculate_distance(yolo_center[0], yolo_center[1], hand_center[0], hand_center[1])
                output['distance'] = distance

                print(distance)
                output['Prohibited Item Use'] = False
                if distance < 200:
                    output['Prohibited Item Use'] = True
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    #cv2.putText(frame, f"{int(distance)}cm", (xmax - 80, ymin - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
                #cv2.line(frame, (int(yolo_center[0]), int(yolo_center[1])), (int(hand_center[0]), int(hand_center[1])), (0, 255, 0), 2)
                #cv2.putText(frame, f"{int(distance)/7.5}cm", (xmax - 80, ymin - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    output['illegal_objects'] = len(yolo_bboxes)


    return output
    
        # Display the frame with annotations
    
def inference(frame):
    # Load the image
    initialize()

    # Run your functions
    output = handYOLO(frame)

    # Print the output
    return output    
def main(image_path):
    # Load the image
    frame = cv2.imread(image_path)

    # Initialize the global variables
    initialize()

    # Run your functions
    output = handYOLO(frame)

    # Print the output
    print(output)

if __name__ == "__main__":
    main('test.png')
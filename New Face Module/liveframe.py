def getGoodCapture(src_image):
    import cv2
    from deepface import DeepFace
    main_src = cv2.imread(src_image)
    
    isVerified = False
    while not isVerified:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error in capturing the frame")
                break
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.verify(img1_path=frame_new, img2_path=src_image, detector_backend='mediapipe', model_name='ArcFace')
        print(result)
        isVerified = result['verified']
    
    cv2.imwrite("VerifiedLiveFrame.jpg", frame)

if __name__ == "__main__":
    src = input("Enter the source image path: ")
    getGoodCapture(src)
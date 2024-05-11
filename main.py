import cv2

def face_detection(save_path='mohammed.jpg'):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return 0

    num_faces_detected = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # If a face is detected, save the photo
        if len(faces) > 0:
            cv2.imwrite(save_path, frame)
            print("Face detected! Photo saved as", save_path)
            num_faces_detected = len(faces)
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

    return num_faces_detected

num_faces = face_detection()
print("Number of faces detected:", num_faces)

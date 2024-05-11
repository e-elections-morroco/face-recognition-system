import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load existing user images and their names
existing_users = {
    # "Ouail": cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE),
    "mohammed": cv2.imread("mohammed.jpg", cv2.IMREAD_GRAYSCALE),
}

# Capture an image from the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Convert the captured frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale frame
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate through each face
for (x, y, w, h) in faces:
    # Extract the face region
    face_roi = gray[y:y+h, x:x+w]
    
    # Compare the detected face with existing users
    for name, user_image in existing_users.items():
        # Resize existing user image to match the size of detected face
        user_image_resized = cv2.resize(user_image, (w, h))
        
        # Compare the two images using some algorithm (e.g., mean squared error)
        mse = ((face_roi - user_image_resized) ** 2).mean()
        
        # If the mean squared error is below a threshold, consider it a match
        if mse < 1000:
            print("User:", name)
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()

import cv2
from tkinter import Tk, Label, Button, Entry, StringVar
from deepface import DeepFace
import os
import face_recognition

# Global Variables
user_name = ""
images_captured = 0
dataset_path = "user_datasets"

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)


def home_page():
    """Create the home page."""
    home = Tk()
    home.title("Face Recognizer")

    Label(home, text="Face Recognizer", font=("Arial", 20)).pack(pady=20)
    Button(home, text="Add a User", width=20, command=lambda: add_user_page(home)).pack(pady=10)
    Button(home, text="Check a User", width=20, command=lambda: check_user_page(home)).pack(pady=10)

    home.mainloop()


def add_user_page(home):
    """Create the add user page."""
    home.destroy()
    add_user = Tk()
    add_user.title("Add a User")

    Label(add_user, text="Enter User Name:", font=("Arial", 14)).pack(pady=10)
    name_var = StringVar()
    Entry(add_user, textvariable=name_var, font=("Arial", 14)).pack(pady=10)
    Button(
        add_user,
        text="Next",
        width=20,
        command=lambda: capture_dataset_page(add_user, name_var.get()),
    ).pack(pady=10)

    add_user.mainloop()


def capture_dataset_page(add_user, name):
    """Create the capture dataset page."""
    global user_name, images_captured
    add_user.destroy()
    user_name = name
    images_captured = 0

    capture_page = Tk()
    capture_page.title("Capture Dataset")

    Label(capture_page, text=f"User: {user_name}", font=("Arial", 14)).pack(pady=10)
    count_label = Label(capture_page, text=f"Number of images captured = {images_captured}", font=("Arial", 14))
    count_label.pack(pady=10)
    Button(capture_page, text="Capture Dataset", width=20, command=lambda: capture_images(count_label)).pack(pady=10)
    Button(capture_page, text="Train the Model", width=20, command=lambda: train_model()).pack(pady=10)

    capture_page.mainloop()


def capture_images(count_label):
    """Capture images of the user."""
    global images_captured, user_name
    user_folder = os.path.join(dataset_path, user_name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while images_captured < 300:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            # Save image
            image_path = os.path.join(user_folder, f"{images_captured}.jpg")
            cv2.imwrite(image_path, face_roi)

            # Emotion analysis
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Display information
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{user_name} is {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            images_captured += 1
            count_label.config(text=f"Number of images captured = {images_captured}")

        cv2.imshow("Capturing Images", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model():
    """Train the model (Placeholder for now)."""
    print("Training the model... (Functionality to be implemented)")


def check_user_page(home):
    """Check if a user exists."""
    home.destroy()
    check_user = Tk()
    check_user.title("Check a User")

    Label(check_user, text="Checking User", font=("Arial", 14)).pack(pady=10)

    Button(
        check_user,
        text="Start Camera",
        width=20,
        command=lambda: recognize_user(),
    ).pack(pady=10)

    check_user.mainloop()


def recognize_user():
    """Recognize the user."""
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    known_encodings = []
    known_names = []

    # Load known face encodings
    for user_folder in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_folder)
        for img_file in os.listdir(user_path):
            img_path = os.path.join(user_path, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(user_folder)

    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Recognizing User", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Start the application
home_page() 
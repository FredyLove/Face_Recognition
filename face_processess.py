import cv2
import os

# Paths
RAW_DATASET_PATH = "dataset/lfw_funneled"  # or the extracted folder name
PROCESSED_PATH = "processed_dataset"

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create processed folder if it doesn't exist
if not os.path.exists(PROCESSED_PATH):
    os.makedirs(PROCESSED_PATH)

# Loop through all person folders
for person_name in os.listdir(RAW_DATASET_PATH):
    person_path = os.path.join(RAW_DATASET_PATH, person_name)
    
    if not os.path.isdir(person_path):
        continue  # Skip if it's not a folder

    # Create a folder in the processed dataset
    processed_person_path = os.path.join(PROCESSED_PATH, person_name)
    os.makedirs(processed_person_path, exist_ok=True)

    # Process each image in person's folder
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Save only the first detected face
        for i, (x, y, w, h) in enumerate(faces):
            face_crop = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (160, 160))  # You can change size based on your model

            save_path = os.path.join(processed_person_path, f"{img_name}")
            cv2.imwrite(save_path, face_resized)
            break  # Only save one face per image

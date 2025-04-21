import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and label map
model = load_model("face_recognition_model.h5")

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

IMG_SIZE = 160
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result_img = img.copy()
    predictions_text = ""

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        predictions = model.predict(face)[0]
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx]) * 100
        label = label_map[pred_idx]

        predictions_text += f"{label} ({confidence:.2f}%)\n"

        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_img, f"{label} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return result_img, predictions_text if predictions_text else "No face detected."

# ---------------------------
# GUI Functionality
# ---------------------------
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    predicted_img, result_text = predict_image(file_path)
    rgb_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_img)
    img_pil = img_pil.resize((500, 400))  # Resize for display
    img_tk = ImageTk.PhotoImage(img_pil)

    image_panel.config(image=img_tk)
    image_panel.image = img_tk
    result_label.config(text=result_text)

# ---------------------------
# GUI Setup
# ---------------------------
root = tk.Tk()
root.title("🔍 AI Face Recognition System")

# Theme
style = ttk.Style()
style.theme_use("clam")

# Frame Setup
frame = ttk.Frame(root, padding=20)
frame.grid(row=0, column=0)

# Title
ttk.Label(frame, text="📷 Upload an Image to Recognize a Face", font=("Segoe UI", 16)).grid(row=0, column=0, pady=(0, 10))

# Upload Button
upload_btn = ttk.Button(frame, text="Upload Image", command=upload_image)
upload_btn.grid(row=1, column=0, pady=10)

# Image Display Panel
image_panel = tk.Label(frame)
image_panel.grid(row=2, column=0, pady=10)

# Result Label
result_label = ttk.Label(frame, text="", font=("Segoe UI", 12), foreground="green", anchor="center", justify="center")
result_label.grid(row=3, column=0, pady=10)

# Run the app
root.mainloop()

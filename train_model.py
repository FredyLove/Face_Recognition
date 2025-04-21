import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# -------------------------------
# Step 1: Load and Preprocess Dataset
# -------------------------------

DATASET_PATH = "Data/Archive/processed_dataset"
IMG_SIZE = 160  # You can lower this to 96 or 64 if needed

X = []
y = []
label_map = {}
label_id = 0

print("🔍 Loading dataset from:", DATASET_PATH)

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0

            if img.shape != (IMG_SIZE, IMG_SIZE, 3):
                print(f"❌ Bad image shape at {img_path}: {img.shape}")
                continue

            X.append(img)
            y.append(label_id)

        except Exception as e:
            print(f"⚠️ Error reading {img_path}: {e}")

    label_id += 1

X = np.array(X)
y = np.array(y)

print("\n✅ Dataset Summary:")
print("X shape:", X.shape)         # e.g., (1000, 160, 160, 3)
print("y shape:", y.shape)         # e.g., (1000,)
print("Classes:", label_map)

# One-hot encode labels
y_cat = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# -------------------------------
# Step 2: Define CNN Model
# -------------------------------

def create_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Build and compile the model
model = create_model((IMG_SIZE, IMG_SIZE, 3), y_cat.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Starting training...")

# -------------------------------
# Step 3: Train the Model
# -------------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# -------------------------------
# Step 4: Save Model and Labels
# -------------------------------

model.save("face_recognition_model.h5")
print("✅ Model saved to face_recognition_model.h5")

# Save label map
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)
print("✅ Label map saved to label_map.pkl")

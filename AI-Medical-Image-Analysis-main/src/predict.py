import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 64


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Image not found")

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    flattened = normalized.flatten().reshape(1, -1)

    return image, flattened


def predict_image(image_path):
    model_path = "models/medical_model.pkl"

    if not os.path.exists(model_path):
        print("Model not found.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    original, processed = preprocess_image(image_path)

    prediction = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0]

    class_name = "PNEUMONIA" if prediction == 1 else "NORMAL"
    confidence = max(prob) * 100

    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.2f}%")

    plt.imshow(original, cmap="gray")
    plt.title(f"{class_name} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
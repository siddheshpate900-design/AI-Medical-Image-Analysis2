import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

from src.data_loader import load_dataset


def evaluate_model():
    model_path = "models/medical_model.pkl"

    if not os.path.exists(model_path):
        print("Model not found.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    _, _, _, _, X_test, y_test = load_dataset()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["NORMAL", "PNEUMONIA"])
    disp.plot()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()
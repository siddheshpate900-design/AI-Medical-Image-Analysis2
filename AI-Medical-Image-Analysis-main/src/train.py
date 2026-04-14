import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.data_loader import load_dataset
from src.model import build_model


def train_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Test samples:", len(X_test))

    if len(X_train) == 0:
        print("No training data found.")
        return

    model = build_model()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred) if len(X_val) > 0 else 0

    print(f"Validation Accuracy: {val_acc:.4f}")

    with open("models/medical_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Saved model to models/medical_model.pkl")

    plt.figure(figsize=(6, 4))
    plt.bar(["Validation Accuracy"], [val_acc])
    plt.ylim(0, 1)
    plt.title("Validation Accuracy")
    plt.savefig("outputs/training_result.png")
    plt.show()
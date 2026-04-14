import os
import cv2
import numpy as np

IMG_SIZE = 64


def load_images_from_folder(folder, label):
    data = []
    labels = []

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return data, labels

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        image = image.flatten()

        data.append(image)
        labels.append(label)

    return data, labels


def load_dataset(base_path="data"):
    train_data = []
    train_labels = []

    val_data = []
    val_labels = []

    test_data = []
    test_labels = []

    # Train
    normal_train, normal_train_labels = load_images_from_folder(
        os.path.join(base_path, "train", "NORMAL"), 0
    )
    pneumonia_train, pneumonia_train_labels = load_images_from_folder(
        os.path.join(base_path, "train", "PNEUMONIA"), 1
    )

    train_data.extend(normal_train)
    train_data.extend(pneumonia_train)
    train_labels.extend(normal_train_labels)
    train_labels.extend(pneumonia_train_labels)

    # Validation
    normal_val, normal_val_labels = load_images_from_folder(
        os.path.join(base_path, "val", "NORMAL"), 0
    )
    pneumonia_val, pneumonia_val_labels = load_images_from_folder(
        os.path.join(base_path, "val", "PNEUMONIA"), 1
    )

    val_data.extend(normal_val)
    val_data.extend(pneumonia_val)
    val_labels.extend(normal_val_labels)
    val_labels.extend(pneumonia_val_labels)

    # Test
    normal_test, normal_test_labels = load_images_from_folder(
        os.path.join(base_path, "test", "NORMAL"), 0
    )
    pneumonia_test, pneumonia_test_labels = load_images_from_folder(
        os.path.join(base_path, "test", "PNEUMONIA"), 1
    )

    test_data.extend(normal_test)
    test_data.extend(pneumonia_test)
    test_labels.extend(normal_test_labels)
    test_labels.extend(pneumonia_test_labels)

    return (
        np.array(train_data), np.array(train_labels),
        np.array(val_data), np.array(val_labels),
        np.array(test_data), np.array(test_labels)
    )
import os


def print_menu():
    print("\nAI-Powered Medical Image Analysis")
    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Predict Single Image")
    print("4. Exit")


def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            from src.train import train_model
            train_model()

        elif choice == "2":
            from src.evaluate import evaluate_model
            evaluate_model()

        elif choice == "3":
            image_path = input("Enter image path: ").strip()
            if not os.path.exists(image_path):
                print("Image path not found.")
                continue

            from src.predict import predict_image
            predict_image(image_path)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
if __name__ == "__main__":
    main()
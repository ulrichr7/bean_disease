import matplotlib.pyplot as plt

def plot_history(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Training")

    plt.plot(history.history["val_accuracy"], label="Validation")

    plt.legend()

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.show()
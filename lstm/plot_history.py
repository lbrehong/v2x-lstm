import matplotlib.pyplot as plt
import json
import os

from config import OUTPUT_DIR


def plot_losses(history, title="Model"):
    """Plot Training & Validation Loss."""
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} Training vs Validation Loss')
    plt.show()


if __name__ == "__main__":
    RAT = input("Enter the RAT (5g/pc5/dsrc): ")

    with open(os.path.join(OUTPUT_DIR, f"lstm_{RAT}_training_history.json"), "r") as f:
        history_lstm = json.load(f)
    with open(os.path.join(OUTPUT_DIR, f"gru_{RAT}_training_history.json"), "r") as f:
        history_gru = json.load(f)
    with open(os.path.join(OUTPUT_DIR, f"rnn_{RAT}_training_history.json"), "r") as f:
        history_rnn = json.load(f)

    plot_losses(history_lstm, "LSTM")
    plot_losses(history_gru, "GRU")
    plot_losses(history_rnn, "RNN")

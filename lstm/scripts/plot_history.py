"""
Training history visualization utility.

Loads saved training history JSON files and plots training vs validation
loss curves for visual analysis of model convergence.

Usage:
    python -m scripts.plot_history
    # Enter RAT when prompted: 5g, pc5, or dsrc
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import json

from config import OUTPUT_DIR


def plot_losses(history, title="Model"):
    """
    Plot training and validation loss curves.

    Creates a line plot comparing training loss against validation loss
    across epochs, useful for detecting overfitting.

    Args:
        history: Dictionary with 'loss' and 'val_loss' keys containing
                 lists of loss values per epoch
        title: Model name for plot title (default: "Model")
    """
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} Training vs Validation Loss')
    plt.show()


if __name__ == "__main__":
    # Interactive prompt for RAT selection
    RAT = input("Enter the RAT (5g/pc5/dsrc): ")

    # Load training history for all three model architectures
    with open(os.path.join(OUTPUT_DIR, f"lstm_{RAT}_training_history.json"), "r") as f:
        history_lstm = json.load(f)
    with open(os.path.join(OUTPUT_DIR, f"gru_{RAT}_training_history.json"), "r") as f:
        history_gru = json.load(f)
    with open(os.path.join(OUTPUT_DIR, f"rnn_{RAT}_training_history.json"), "r") as f:
        history_rnn = json.load(f)

    # Display loss plots for each architecture
    plot_losses(history_lstm, "LSTM")
    plot_losses(history_gru, "GRU")
    plot_losses(history_rnn, "SimpleRNN")

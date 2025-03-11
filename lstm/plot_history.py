import matplotlib.pyplot as plt
import json

def plot_losses(history):
    # Plot Training & Validation Loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Training vs Validation Loss')
    plt.show()


RAT = input("Enter the RAT: ")


with open("lstm_" + RAT + "_training_history.json", "r") as f:
    history_lstm = json.load(f)
with open("gru_" + RAT + "_training_history.json", "r") as f:
    history_gru = json.load(f)
with open("rnn_" + RAT + "_training_history.json", "r") as f:
    history_rnn = json.load(f)

# Plot losses
plot_losses(history_lstm)
plot_losses(history_gru)
plot_losses(history_rnn)
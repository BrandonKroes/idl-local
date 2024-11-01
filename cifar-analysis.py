import json
import matplotlib.pyplot as plt

# Load results from JSON files
with open('cifar10_mlp_results.json', 'r') as f:
    mlp_results = json.load(f)
with open('cifar10_cnn_results.json', 'r') as f:
    cnn_results = json.load(f)

# Function to plot accuracy over epochs for each configuration
def plot_accuracy(results, model_name):
    plt.figure(figsize=(10, 6), dpi=300)
    
    for config in results:
        epochs = range(1, len(config['epochs']) + 1)
        train_acc = [epoch['accuracy'] for epoch in config['epochs']]
        val_acc = [epoch['val_accuracy'] for epoch in config['epochs']]
        
        # Label showing optimizer, activation, and regularizer
        label = (f"Optimizer: {config['optimizer']}, "
                 f"Activation: {config['activation']}, "
                 f"Regularizer: {config['regularizer']}")
        
        # Plot both training and validation accuracy
        plt.plot(epochs, train_acc, label=f"Train - {label}")
        plt.plot(epochs, val_acc, linestyle='--', label=f"Validation - {label}")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Training and Validation Accuracy on CIFAR-10")
    plt.legend(loc="best", fontsize='small')
    plt.grid(True)
    
    # Save plot to a PNG file
    plt.savefig(f"{model_name.lower()}_accuracy_plot.png")
    plt.show()

# Plot for MLP results
plot_accuracy(mlp_results, "MLP")

# Plot for CNN results
plot_accuracy(cnn_results, "CNN")

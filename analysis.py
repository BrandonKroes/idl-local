import json
import matplotlib.pyplot as plt

# Load MLP, CNN, and CIFAR-10 results from JSON files
with open('mlp_fashion_mnist_results.json', 'r') as file:
    mlp_results = json.load(file)
with open('cnn_fashion_mnist_results.json', 'r') as file:
    cnn_results = json.load(file)
# Function to plot learning curves for each configuration set
def plot_results(results, title, filename_prefix):
    plt.figure(figsize=(10, 6), dpi=300)
    for config in results:
        epochs = list(range(1, len(config['epochs']) + 1))
        val_accuracies = [epoch_data['val_accuracy'] for epoch_data in config['epochs']]
        val_losses = [epoch_data['val_loss'] for epoch_data in config['epochs']]
        
        # Label for the configuration
        label = f"Activation: {config['activation']}, Optimizer: {config['optimizer']}, Regularizer: {config['regularizer']}"

        # Plot validation accuracy and loss
        plt.plot(epochs, val_accuracies, label=f"Acc - {label}", linestyle='-', marker='o')
        plt.plot(epochs, val_losses, label=f"Loss - {label}", linestyle='--', marker='x')

    # Graph details
    plt.title(f'{title} Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy / Loss')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    
    # Save graph as a high-quality image
    plt.savefig(f'{filename_prefix}_learning_curves.png')
    plt.close()
    print(f"{title} plot saved as {filename_prefix}_learning_curves.png")

# Plot learning curves for MLP, CNN, and CIFAR-10 results
plot_results(mlp_results, 'MLP on Fashion MNIST', 'mlp_fashion_mnist')
plot_results(cnn_results, 'CNN on Fashion MNIST', 'cnn_fashion_mnist')
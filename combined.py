
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import json

from tensorflow.keras.datasets import fashion_mnist
import json
import itertools

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Parameters for experimentation
activations = ['relu', 'elu', 'selu']
optimizer_classes = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}
regularizers = {'l1': l1(0.01), 'l2': l2(0.01), 'l1_l2': l1_l2(0.01, 0.01)}

# Storage for results
results = []

# Define a function to create and compile the MLP model
def create_mlp_model(activation, optimizer_class, regularizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation=activation, kernel_regularizer=regularizer),
        Dense(128, activation=activation, kernel_regularizer=regularizer),
        Dense(10, activation='softmax')
    ])
    # Initialize a fresh optimizer instance for each model
    optimizer = optimizer_class()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Run all combinations of hyperparameters
for activation, (opt_name, optimizer_class), (reg_name, regularizer) in itertools.product(activations, optimizer_classes.items(), regularizers.items()):
    print(f"Training MLP with activation={activation}, optimizer={opt_name}, regularizer={reg_name}")
    model = create_mlp_model(activation, optimizer_class, regularizer)
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=0)

    # Save each combination's result
    results.append({
        'activation': activation,
        'optimizer': opt_name,
        'regularizer': reg_name,
        'epochs': [
            {'val_accuracy': float(acc), 'val_loss': float(loss), 'accuracy': float(train_acc)}
            for acc, loss, train_acc in zip(history.history['val_accuracy'], history.history['val_loss'], history.history['accuracy'])
        ]
    })

# Save all results to a JSON file
with open('mlp_fashion_mnist_results.json', 'w') as file:
    json.dump(results, file, indent=4)

print("MLP training complete and results saved to mlp_fashion_mnist_results.json")



# Load Fashion MNIST data, reshaping for CNN input
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Parameters for experimentation
activations = ['relu', 'elu', 'selu']
optimizer_classes = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}
regularizers = {'l1': l1(0.01), 'l2': l2(0.01), 'l1_l2': l1_l2(0.01, 0.01)}

# Storage for results
results = []

# Define a function to create and compile the CNN model
def create_cnn_model(activation, optimizer_class, regularizer):
    model = Sequential([
        Input(shape=(28, 28, 1)),  # Define input shape using Input layer
        Conv2D(32, (3, 3), activation=activation, kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=activation, kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation=activation, kernel_regularizer=regularizer),
        Dense(10, activation='softmax')
    ])
    # Create a new optimizer instance for each model run
    optimizer = optimizer_class()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Run all combinations of hyperparameters
for activation, (opt_name, optimizer_class), (reg_name, regularizer) in itertools.product(activations, optimizer_classes.items(), regularizers.items()):
    print(f"Training CNN with activation={activation}, optimizer={opt_name}, regularizer={reg_name}")
    model = create_cnn_model(activation, optimizer_class, regularizer)
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=0)

    # Save each combination's result
    results.append({
        'activation': activation,
        'optimizer': opt_name,
        'regularizer': reg_name,
        'epochs': [
            {'val_accuracy': float(acc), 'val_loss': float(loss), 'accuracy': float(train_acc)}
            for acc, loss, train_acc in zip(history.history['val_accuracy'], history.history['val_loss'], history.history['accuracy'])
        ]
    })

# Save all results to a JSON file
with open('cnn_fashion_mnist_results.json', 'w') as file:
    json.dump(results, file, indent=4)

print("CNN training complete and results saved to cnn_fashion_mnist_results.json")



# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

# Load MLP and CNN results from JSON files
with open('mlp_fashion_mnist_results.json', 'r') as f:
    mlp_results = json.load(f)
with open('cnn_fashion_mnist_results.json', 'r') as f:
    cnn_results = json.load(f)

# Helper function to get the top 3 configurations based on validation accuracy
def get_top_configurations(results, top_n=3):
    # Sort by the last validation accuracy in training epochs
    sorted_results = sorted(results, key=lambda x: x['epochs'][-1]['val_accuracy'], reverse=True)[:top_n]
    top_configs = []
    for config in sorted_results:
        # Retrieve optimizer class from string name
        if config['optimizer'] == 'adam':
            optimizer = Adam
        elif config['optimizer'] == 'sgd':
            optimizer = SGD
        elif config['optimizer'] == 'rmsprop':
            optimizer = RMSprop
        else:
            continue
        # Retrieve regularizer instance from string name
        regularizer = None
        if config['regularizer'] == 'l1':
            regularizer = l1(0.01)
        elif config['regularizer'] == 'l2':
            regularizer = l2(0.01)
        elif config['regularizer'] == 'l1_l2':
            regularizer = l1_l2(0.01, 0.01)
        
        # Add config dictionary for each model
        top_configs.append({
            'activation': config['activation'],
            'optimizer': optimizer,
            'regularizer': regularizer
        })
    return top_configs

# Get the top 3 MLP and top 3 CNN configurations
top_mlp_configs = get_top_configurations(mlp_results, top_n=3)
top_cnn_configs = get_top_configurations(cnn_results, top_n=3)

# Define function to create an MLP model
def create_mlp_models(activation, optimizer_class, regularizer):
    model = Sequential([
        Input(shape=(32*32*3,)),
        Dense(256, activation=activation, kernel_regularizer=regularizer),
        Dense(128, activation=activation, kernel_regularizer=regularizer),
        Dense(10, activation='softmax')
    ])
    optimizer = optimizer_class()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define function to create a CNN model
def create_cnn_model(activation, optimizer_class, regularizer):
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation=activation, kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=activation, kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation=activation, kernel_regularizer=regularizer),
        Dense(10, activation='softmax')
    ])
    optimizer = optimizer_class()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train top MLP configurations on CIFAR-10
x_train_flat = x_train.reshape(-1, 32*32*3)
x_test_flat = x_test.reshape(-1, 32*32*3)
mlp_results = []
for config in top_mlp_configs:
    print(f"Training MLP on CIFAR-10 with activation={config['activation']}, optimizer={config['optimizer'].__name__}, regularizer={config['regularizer']}")
    model = create_mlp_models(config['activation'], config['optimizer'], config['regularizer'])
    history = model.fit(x_train_flat, y_train, epochs=100, validation_data=(x_test_flat, y_test), verbose=0)
    mlp_results.append({
        'activation': config['activation'],
        'optimizer': config['optimizer'].__name__,
        'regularizer': str(config['regularizer']),
        'epochs': [
            {'val_accuracy': float(acc), 'val_loss': float(loss), 'accuracy': float(train_acc)}
            for acc, loss, train_acc in zip(history.history['val_accuracy'], history.history['val_loss'], history.history['accuracy'])
        ]
    })

# Train top CNN configurations on CIFAR-10
cnn_results = []
for config in top_cnn_configs:
    print(f"Training CNN on CIFAR-10 with activation={config['activation']}, optimizer={config['optimizer'].__name__}, regularizer={config['regularizer']}")
    model = create_cnn_model(config['activation'], config['optimizer'], config['regularizer'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=0)
    cnn_results.append({
        'activation': config['activation'],
        'optimizer': config['optimizer'].__name__,
        'regularizer': str(config['regularizer']),
        'epochs': [
            {'val_accuracy': float(acc), 'val_loss': float(loss), 'accuracy': float(train_acc)}
            for acc, loss, train_acc in zip(history.history['val_accuracy'], history.history['val_loss'], history.history['accuracy'])
        ]
    })

# Save results to JSON
with open('cifar10_mlp_results.json', 'w') as file:
    json.dump(mlp_results, file, indent=4)
with open('cifar10_cnn_results.json', 'w') as file:
    json.dump(cnn_results, file, indent=4)

print("CIFAR-10 MLP and CNN training complete. Results saved to cifar10_mlp_results.json and cifar10_cnn_results.json.")
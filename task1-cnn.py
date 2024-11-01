import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import json
import itertools

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
    history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test), verbose=0)

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

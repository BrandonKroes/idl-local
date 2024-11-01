import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
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
with open('mlp_fashion_mnist_results.json', 'w') as file:
    json.dump(results, file, indent=4)

print("MLP training complete and results saved to mlp_fashion_mnist_results.json")

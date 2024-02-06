
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

initial_weights = None
final_weights = None
weight_evolution = []
norm_error_epochs = []
num_epochs = 0
permissible_error = 0

def train_perceptron(learning_rate, epochs, file_path, progress_bar=None):
    global norm_error_epochs, weight_evolution, initial_weights, final_weights, num_epochs

    norm_error_epochs.clear()
    weight_evolution.clear()

    delimiter = ';'
    
    dataframe = pd.read_csv(file_path, delimiter=delimiter, header=None)

    num_features = len(dataframe.columns) - 1

    weights = np.random.uniform(low=0, high=1, size=(num_features + 1, 1)).round(4)
    input_columns = np.hstack([dataframe.iloc[:, :-1].values, np.ones((dataframe.shape[0], 1))])
    output_column = np.array(dataframe.iloc[:, -1])

    initial_weights = weights.copy()
    num_epochs = epochs

    print(f"Initial weights: {weights}")

    for _ in range(num_features + 1):
        weight_evolution.append([])

    for epoch in range(epochs):
        u = np.dot(input_columns, weights)
        predicted_output = np.where(u >= 0, 1, 0).reshape(-1, 1)
        errors = output_column.reshape(-1, 1) - predicted_output

        norm_error = np.linalg.norm(errors)
        norm_error_epochs.append(norm_error)

        for i in range(num_features + 1):
            weight_evolution[i].append(weights[i, 0])

        errors_product = np.dot(input_columns.T, errors)
        delta_w = learning_rate * errors_product
        weights += delta_w

        if progress_bar:
            # Update progress bar if provided
            progress_bar['value'] = (epoch + 1) / epochs * 100
            progress_bar.update()

    final_weights = weights

def show_results():
    global norm_error_epochs, weight_evolution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(norm_error_epochs) + 1), norm_error_epochs)
    plt.title('Error Norm Evolution (|e|)')
    plt.xlabel('Epoch')
    plt.ylabel('Error Norm')

    plt.subplot(1, 2, 2)
    for i, epoch_weights in enumerate(weight_evolution[:-1]):  # Exclude the last weight
        plt.plot(range(1, len(epoch_weights) + 1), epoch_weights, label=f'Weight {i + 1}')
    plt.title('Weight Value Evolution (W)')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_weights():
    return initial_weights, final_weights, num_epochs, permissible_error

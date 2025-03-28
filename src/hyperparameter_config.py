import LSTM
from aitaDataset import training_data, test_data
import cross_validation
import torch.nn as nn
import torch

num_layers = [1, 2]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
hidden_sizes = [10 * (i+1) for i in range(10)]
training_epochs = [i+1 for i in range(2)]

models = []
criterion = nn.MSELoss() # mean squared error for the loss function
optimizers = []

for layer in num_layers:
    for hidden_size in hidden_sizes:
        model = LSTM.LSTM(input_size=1140, hidden_dim=hidden_size, layer_dim=layer, output_dim=1)
        models.append(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizers.append(optimizer)

best_model, best_num_epochs, best_batch_size, scores = cross_validation.kFoldCrossValidation(training_data, 5, models[:1], training_epochs, batch_sizes[:2], criterion, optimizers[:1])

print(f"Best model found was {best_model} using {best_num_epochs} training rounds and batch size of {best_batch_size}\n\n")
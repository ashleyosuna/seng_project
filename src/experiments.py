from aitaDataset import training_data, test_data
import LSTM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

model = LSTM.LSTM(input_size=1140, hidden_dim=100, layer_dim=1, output_dim=1)
criterion = nn.MSELoss() # mean squared error for the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# VARYING BATCH SIZES
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
train_loss_logger = []
test_loss_logger = []

for size in batch_sizes:
  print(f"Batch size {size}\n---------")
  train_dataloader = DataLoader(training_data, batch_size=size, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size=size, shuffle=True)

  train_loss = LSTM.train_model(train_dataloader, model, criterion, optimizer=optimizer)
  test_loss = LSTM.test_model(test_dataloader, model, criterion)

  train_loss_logger.append(train_loss)
  test_loss_logger.append(test_loss)
  print()

plt.plot(batch_sizes, train_loss_logger, label="Training loss")
plt.plot(batch_sizes, test_loss_logger, label="Test loss")
plt.xlabel("Batch Size")
plt.ylabel("Loss")
plt.title("Mean Squared Error over different Batch Sizes")
plt.legend()
plt.show()
plt.close()
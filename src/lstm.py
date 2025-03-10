# code modified from https://www.geeksforgeeks.org/long-short-term-memory-networks-using-pytorch/

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_df = pd.read_csv('processed.csv')
data_df = data_df.sample(frac = 1) # shuffle rows

X = data_df.iloc[:, :-1]
y = data_df.iloc[:, -1]
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
testYcpy = testY

trainX = torch.tensor(trainX.values, dtype=torch.float32)
trainX = trainX.unsqueeze(1) # reshaping so it has [batch_size, sequence_length, input_size], which is needed by the model
testX = torch.tensor(testX.values, dtype=torch.float32)
testX = testX.unsqueeze(1) 
print(trainX.size())

trainY = torch.tensor(trainY.values, dtype=torch.float32)
trainY = trainY.unsqueeze(-1)
testY = torch.tensor(testY.values, dtype=torch.float32)
testY = testY.unsqueeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        '''
        If the initial hidden state/short term memory (h0) and initial cell state/long term memory (c0) are not provided, 
        they are initialized as zero tensors with shapes [layer_dim, batch_size, hidden_dim]

        Returns 
            out: The output after passing through the fully connected layer, which represents the LSTM's prediction or transformed output.
            hn: The hidden state of the LSTM after processing the input sequence.
            cn: The cell state of the LSTM after processing the input sequence.
        '''
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0)) # passes x, h0, c0 through the lstm layer
        out = self.fc(out[:,-1,:]) # selects the output of the last time step
        return out, hn, cn


model = LSTMModel(input_dim=trainX.size(2), hidden_dim=100, layer_dim=1, output_dim=1)
criterion = nn.MSELoss() # mean squared error for the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100 # epochs are number of rounds of training
h0, c0 = None, None

loss_values = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, h0, c0)
    loss = criterion(outputs, trainY)
    loss.backward() # backpropagating the error to update the weights
    optimizer.step()

    h0 = h0.detach() # detach hidden state to avoid backpropagating through the entire history
    c0 = c0.detach() # detach cell state to avoid backpropagating through the entire history

    loss_values.append(loss.item())

    # print loss value every 2 epochs to monitor model's performance
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
test_outputs, _, _ = model(testX) # only one forward pass, no backpropagation to update weights anymore
mse_loss = nn.MSELoss()
test_outputs.size()
test_mse = mse_loss(test_outputs, testY)
print(f"Mean Squared Error on the Test Set: {test_mse}")

plt.figure(figsize=(12, 6))
plt.plot(range(200), testYcpy, label='Original Data')
plt.plot(range(200), test_outputs.detach().numpy(), label='Predicted Data', linestyle='--')
plt.title('LSTM Model Predictions vs. Original Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.close()

# plotting the training loss
plt.plot(range(1, num_epochs + 1), loss_values, color='b', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Training Loss over Epochs')
plt.legend()
plt.show()
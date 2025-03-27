import torch.nn as nn
import torch

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_dim, layer_dim, output_dim=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, layer_dim, batch_first=True)
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
      h0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim).requires_grad_()
      c0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim).requires_grad_()
            
    out, (hn, cn) = self.lstm(x, (h0, c0)) # passes x, h0, c0 through the lstm layer
    out = self.fc(out[:,-1,:]) # selects the output of the last time step
    return out, hn, cn
  
def train_model(data_loader, model, loss_function, optimizer):
  num_batches = len(data_loader)
  total_loss = 0

  model.train()
  optimizer.zero_grad()

  for X, y in data_loader:
    X = X.float()
    X = X.unsqueeze(1)
    y = y.to(torch.float32)
    outputs, h0, c0 = model(X)
    # loss = criterion(output.squeeze(1), y)
    loss = loss_function(outputs.squeeze(1), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  avg_loss = total_loss / num_batches
  print(f"Train loss: {avg_loss}")
  return avg_loss

def test_model(data_loader, model, loss_function):
  num_batches = len(data_loader)
  total_loss = 0

  model.eval()
  with torch.no_grad():
      for X, y in data_loader:
          X = X.float()
          X = X.unsqueeze(1)
          outputs, h0, c0 = model(X)
          total_loss += loss_function(outputs.squeeze(1), y).item()

  avg_loss = total_loss / num_batches
  print(f"Test loss: {avg_loss}")
  return avg_loss
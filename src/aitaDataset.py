import pandas as pd

class aitaDataset():
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data[idx]
    tfid_vector = row[0:-1]
    label = row[-1]
    return tfid_vector, label
  
data_file = './data/processed_3000.csv'
data = pd.read_csv(data_file, header=None).to_numpy()
num_samples = len(data)
training_data = data[:round(num_samples * 0.8)]
test_data = data[round(num_samples * 0.8):]
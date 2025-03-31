import nltk
from nltk.tokenize import word_tokenize
import re
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import KFold






# Download necessary resources from nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def read_from_csv(filename, X=[], y=[]):

    with open(filename, 'r', newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:  # Skip empty or malformed rows
                continue
            X.append(row[0])  # First column (post content)
            y.append(float(row[1]))  # Second column (sentiment score)

    return X, y

# Specify the directory path
directory = 'data/'

samples = []
labels = []

# Get all file names inside the directory
file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
for file in file_paths:
    samples, labels = read_from_csv(file, samples, labels)


# Limit the size to the first 10,000 rows
samples = samples[:10000]
labels = labels[:10000]


"""
Lemmatization can hurt an LSTM model! I will leave it just in case.
"""
# # Initialize Lemmatizer
# lemmatizer = WordNetLemmatizer()

"""
Breaks common contracted words into their constituent parts.
"""
def decontracted(text):
    # specific
    text = re.sub(r"won[’|']t", "will not", text)
    text = re.sub(r"can[’|']t", "can not", text)

    # general
    text = re.sub(r"n[’|']t", " not", text)
    text = re.sub(r"[’|']re", " are", text)
    text = re.sub(r"[’|']s", " is", text)
    text = re.sub(r"[’|']d", " would", text)
    text = re.sub(r"[’|']ll", " will", text)
    text = re.sub(r"[’|']t", " not", text)
    text = re.sub(r"[’|']ve", " have", text)
    text = re.sub(r"[’|']m", " am", text)

    text = re.sub(r'\bm(\d+)\b', r'male \1', text, flags=re.IGNORECASE)  # Replace 'm<number>'
    text = re.sub(r'\bf(\d+)\b', r'female \1', text, flags=re.IGNORECASE)  # Replace 'f<number>'
    return text

"""
Preprocessing function
"""
def clean_text(text):
    # Remove leading/trailing whitespace
    text.strip()

    # Lowercase the text
    text = text.lower()

    # Decontract text
    text = decontracted(text)
    
    # Remove unwanted characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove zero-width characters
    text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)

    # # Remove numbers
    # text = re.sub(r'\d+', '', text)
    
    return text

# Lemmatizer
lemmatizer = WordNetLemmatizer()

"""
Lemmatizes and tokenizes text. Also removes stopwords during tokenization.
"""
def lemmatize_text(text):
    # Tokenize
    tokens = word_tokenize(text)
    
    # # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word not in stop_words]
    
    # # Lemmatize each token
    # lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

"""
Builds a dictionary of vocab that maps words to numbers.
"""
def build_vocab(tokens_list):

    all_tokens = [token for tokens in tokens_list for token in tokens]

    # Build vocabulary (word-to-index mapping)
    vocab = {word: idx for idx, word in enumerate(set(all_tokens), 1)}  # Start indices at 1

    # Special value for unknown words at index 0
    vocab['<UNK>'] = 0

    return vocab

def vocab_indices(vocab, tokens):

    # Convert tokens to indices
    indices = [vocab[word] for word in tokens]

    return indices

"""
Pads a list with 0s at the end up until the list reaches max_len
"""
def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        padding_length = max_len - len(seq)
        pad_seq = seq + [0] * padding_length
        return pad_seq
    return seq

# Get the raw tokenized posts
tokenized_posts = []
for post in samples:
    text = clean_text(post)
    tokens = lemmatize_text(text)
    tokenized_posts.append(tokens)

# Create a dictionary for the vocab
vocab = build_vocab(tokens_list=tokenized_posts)

# Turn words into numbers using the vocab dictionary
post_indices = [vocab_indices(vocab, post) for post in tokenized_posts]

# Pad posts indices with 0s to the length of the largest tokenized post
max_length = max(len(sublist) for sublist in post_indices)
padded_post_indices = [pad_sequence(post, max_length) for post in post_indices]

embedding_dim = 100  # (or 50, 200, 300, depending on GloVe file)
vocab_size = len(vocab)

# Load GloVe embeddings into the matrix
glove_path = "glove.6B.100d.txt"  # Change depending on your GloVe version

# Initialize embedding matrix with random numbers for unknown words
embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size + 1, embedding_dim))

# Create a dictionary to hold the GloVe embeddings
glove_dict = {}
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype=np.float32)
        glove_dict[word] = vector

# Fill in the embedding matrix with GloVe vectors
for word, index in vocab.items():
    if word in glove_dict:
        embedding_matrix[index] = glove_dict[word]

print(embedding_matrix.shape)

# Optional: Check how many words were not found in GloVe
not_found_count = sum(1 for word in vocab if word not in glove_dict)
print(f"Words not found in GloVe: {not_found_count}")

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_post_indices, labels, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train_tensor = torch.LongTensor(X_train)  # Shape: (num_train_samples, max_length)
y_train_tensor = torch.FloatTensor(y_train)  # Shape: (num_train_samples,)
X_test_tensor = torch.LongTensor(X_test)  # Shape: (num_test_samples, max_length)
y_test_tensor = torch.FloatTensor(y_test)  # Shape: (num_test_samples,)

# Create a TensorDataset and DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 128  # You can adjust this depending on your memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix, num_layers=1, dropout_prob=0):
        super(LSTMModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout_prob
        
        # Embedding Layer (pre-trained)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Load the pre-trained embedding matrix into the embedding layer
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # Optional: Freeze the embeddings to prevent training them
        self.embedding.weight.requires_grad = False  # Set to True if you want to fine-tune embeddings
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout_prob, batch_first=True, bidirectional=True)
        
        # Fully connected layer (output layer)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass input through embedding layer
        embedded = self.embedding(x)
        
        # Pass embeddings through LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Only take the output from the last time step for regression
        last_hidden_state = h_n[-1]  # shape: (batch_size, hidden_dim)
        
        # Pass the last hidden state through the fully connected layer
        output = self.fc(last_hidden_state)
        
        return output

k = 5
kfold = KFold(n_splits=k)

models = []
train_losses = []
val_losses = []

hidden_dims = [32, 64, 128]
layers = [2, 3]
dropout_list = [0.2, 0.4]

patience = 5

for dim in hidden_dims:
    for layer in layers:
        for drop in dropout_list:
            # Hyperparameters
            vocab_size = len(vocab) + 1  # +1 for padding token
            hidden_dim = dim  # Hidden state size for LSTM
            num_layers = layer
            output_dim = 1  # For regression, output is a single continuous value
            learning_rate = 0.001
            dropout = drop

            # Initialize the LSTM model
            model = LSTMModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, embedding_matrix=embedding_matrix, dropout_prob=dropout)

            # Loss function and optimizer
            criterion = nn.MSELoss()  # Mean Squared Error for regression
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            num_epochs = 50 

            total_train_loss = 0
            total_val_loss = 0

            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                print(f"Fold {fold+1}/{k}")
                
                # Create data loaders for training and validation
                train_subset = Subset(train_dataset, train_idx)
                val_subset = Subset(train_dataset, val_idx)
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

                fold_train_loss = 0.0
                fold_val_loss = 0.0
                best_val_loss = float('inf')
                counter = 0

                for epoch in range(num_epochs):
                    print(f"Epoch [{epoch+1}/{num_epochs}]")
                    model.train()  # Set model to training mode
                    running_loss = 0.0
                    
                    # Training phase
                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        optimizer.zero_grad()  # Zero out previous gradients
                        
                        # Forward pass
                        outputs = model(inputs)  # Shape: (batch_size, 1)
                        
                        # Compute the loss
                        loss = criterion(outputs.squeeze(), labels)  # Squeeze outputs to (batch_size,)
                        running_loss += loss.item()
                        
                        # Backward pass and optimization
                        loss.backward()
                        optimizer.step()

                    # Validation Loss
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            y_val_pred = model(inputs)  # Forward pass
                            loss = criterion(y_val_pred.squeeze(), labels)  # Compute loss
                            val_loss += loss.item()
                    val_loss /= len(val_loader)
                
                    # Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0  # Reset counter
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Early stopping triggered!")
                            break  # Stop training

                # Training Loss
                model.eval()
                fold_train_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in train_loader:
                        y_train_pred = model(inputs)  # Forward pass
                        loss = criterion(y_train_pred.squeeze(), labels)  # Compute loss
                        fold_train_loss += loss.item()
                fold_train_loss /= len(train_loader)

                # Validation Loss
                fold_val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        y_val_pred = model(inputs)  # Forward pass
                        loss = criterion(y_val_pred.squeeze(), labels)  # Compute loss
                        fold_val_loss += loss.item()
                fold_val_loss /= len(val_loader)

                # Accumulate total loss for all folds
                total_train_loss += fold_train_loss
                total_val_loss += fold_val_loss

            # Average the total training and validation losses across all folds
            avg_train_loss = total_train_loss / k
            avg_val_loss = total_val_loss / k

            models.append(model)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Write to file after all folds and epochs are complete
            with open("tuning.txt", 'a') as f:
                f.write(f"Dim: {dim}, Layers: {num_layers}, Epochs: {num_epochs}, Dropout: {drop}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}\n")

best_index = np.argmin(val_losses)
best_model = models[best_index]

# Training Loss (after all epochs)
best_model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        y_test_pred = best_model(inputs)  # Forward pass
        loss = criterion(y_test_pred.squeeze(), labels)  # Compute loss
        test_loss += loss.item()
test_loss /= len(test_loader)
# Write to file after all folds and epochs are complete
with open("tuning.txt", 'a') as f:
    f.write(f'Best Model Test Error: {test_loss}')
torch.save(best_model, 'full_model.pth')
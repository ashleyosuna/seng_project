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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split




# Download necessary resources from nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def read_from_csv(filename):
    X, y = [], []

    with open(filename, 'r', newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:  # Skip empty or malformed rows
                continue
            X.append(row[0])  # First column (post content)
            y.append(float(row[1]))  # Second column (sentiment score)

    return X, y

samples, labels = read_from_csv("data.csv")

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

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    return text

# Lemmatizer
lemmatizer = WordNetLemmatizer()

"""
Lemmatizes and tokenizes text. Also removes stopwords during tokenization.
"""
def lemmatize_text(text):
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
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

embedding_dim = 100  # (or 50, 200, 300, depending on your GloVe file)
vocab_size = len(vocab)

# TODO Look into how to handle missing GloVe words
# Initialize embedding matrix with random numbers for unknown words
embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size + 1, embedding_dim))

# Load GloVe embeddings into the matrix
glove_path = "glove.6B.100d.txt"  # Change depending on your GloVe version

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
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train_tensor = torch.LongTensor(X_train)  # Shape: (num_train_samples, max_length)
y_train_tensor = torch.FloatTensor(y_train)  # Shape: (num_train_samples,)
X_val_tensor = torch.LongTensor(X_val)  # Shape: (num_test_samples, max_length)
y_val_tensor = torch.FloatTensor(y_val)  # Shape: (num_test_samples,)
X_test_tensor = torch.LongTensor(X_test)  # Shape: (num_test_samples, max_length)
y_test_tensor = torch.FloatTensor(y_test)  # Shape: (num_test_samples,)

# Create a TensorDataset and DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32  # You can adjust this depending on your memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super(LSTMModel, self).__init__()
        
        # Embedding Layer (pre-trained)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Load the pre-trained embedding matrix into the embedding layer
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # Optional: Freeze the embeddings to prevent training them
        self.embedding.weight.requires_grad = False  # Set to True if you want to fine-tune embeddings
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        
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

# Hyperparameters
vocab_size = len(vocab) + 1  # +1 for padding token
embedding_dim = 100  # Dimensions of GloVe embeddings
hidden_dim = 64  # Hidden state size for LSTM
output_dim = 1  # For regression, output is a single continuous value
learning_rate = 0.001

# Initialize the LSTM model
model = LSTMModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim, embedding_matrix=embedding_matrix)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 5  # Adjust based on your training needs

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
    
    # Print training statistics every epoch
    print(f"Training Loss: {running_loss / len(train_loader):.4f}")

    #TODO Actually try to get the validation error properly - I don't think this is correct
   # Compute validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
import math
import numpy as np
from torch.utils.data import DataLoader
import aitaDataset
import LSTM

def split_into_k_groups(training_data, k):
    sample_number = training_data.shape[0]

    samples_per_group = math.ceil(sample_number / k )

    X_groups = []

    for i in range(k):
        samples = None
        if i == k - 1:
            samples = training_data[i * samples_per_group:]

        else:
            samples = training_data[i * samples_per_group: i * samples_per_group + samples_per_group]

        X_groups.append(samples)

    return X_groups

def kFoldCrossValidation(training_data, k, algorithms, training_epochs, batch_sizes, criterion, optimizers):
    # split samples into k groups
    X_groups = split_into_k_groups(training_data, k)

    # cross validation
    best_score = float('inf')
    best_model, best_epochs, best_batch_size = None, None, None

    scores = []
    optimizer_index = 0

    for algorithm in algorithms:
        for epochs in training_epochs:
            for batch in batch_sizes:
                print(f"Evaluating model {algorithm}, number of epochs {epochs}, batch_size {batch}\n\n")
                cross_validation_score = 0

                for i in range(len(X_groups)):
                    X_groups_included = [X_groups[k] for k in range(len(X_groups)) if k != i]
                    X_train = np.concatenate(X_groups_included)

                    validation_set = X_groups[i]

                    training_set = aitaDataset.aitaDataset(X_train)
                    validation_set = aitaDataset.aitaDataset(validation_set)

                    train_dataloader = DataLoader(training_set, batch_size=batch, shuffle=True)
                    test_dataloader = DataLoader(validation_set, batch_size=batch, shuffle=True)

                    _ = LSTM.train_model(train_dataloader, algorithm, criterion, optimizers[optimizer_index], epochs)
                    test_loss = LSTM.test_model(test_dataloader, algorithm, criterion)

                    cross_validation_score += test_loss

                cross_validation_score /= k
                scores.append(cross_validation_score)

                if cross_validation_score < best_score:
                    best_score = cross_validation_score
                    best_model, best_epochs, best_batch_size = algorithm, epochs, batch
                        
        optimizer_index += 1

    return best_model, best_epochs, best_batch_size, scores
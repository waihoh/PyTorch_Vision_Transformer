import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from tqdm import tqdm


def train(vision_transformer_model, epochs, learning_rate, dataset):
    optimizer = Adam(params=vision_transformer_model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    losses = []

    for epoch in tqdm(iterable=range(epochs)):
        for x, y in zip(dataset.dataset.data, dataset.dataset.targets):

            x = torch.tensor(x)
            x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
            x = x.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            y_hat = vision_transformer_model(x.float())
            y = torch.tensor([y])
            loss = criterion(y_hat, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    return losses

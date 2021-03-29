from typing import Tuple
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch import optim
from torch import nn
from torch.utils import data
from torch.utils.data.dataloader import DataLoader


def synthetic_data(W: Tensor, b: float, n_examples: int) -> Tuple[Tensor, Tensor]:
    X = torch.normal(0, 1, (n_examples, len(W)))
    y = torch.matmul(X, W) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def create_dataset(data_arrays, batch_size, is_train=True) -> data.DataLoader:
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def create_net(n_inputs, n_outputs):
    return nn.Sequential(
        nn.Linear(n_inputs, n_outputs)
    )

def train(
    net: nn.Module,
    loss: _Loss,
    optimizer: Optimizer,
    train_dataset: DataLoader,
    n_epochs: int = 10,
):

    for epoch in range(n_epochs):
        net.train()
        loss_accum = 0.0
        samples_accum = 0.0
        for X, y in train_dataset:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_accum += l * len(y)
            samples_accum += len(y)
        print(f"Epoch {epoch}: train loss = {(loss_accum / samples_accum):2.2e}")

def main():
    N = 10_000
    BATCH_SIZE = 32
    W_true = torch.tensor([2, -3.4])
    b_true = 4.2

    features, targets = synthetic_data(W_true, b_true, N)
    train_dataset = create_dataset(
        [features, targets],
        batch_size=BATCH_SIZE
    )

    net = create_net(len(W_true), 1)
    loss = nn.MSELoss(reduction="mean")
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    train(net, loss, optimizer, train_dataset)

if __name__ == "__main__":
    main()
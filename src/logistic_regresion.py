from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch import optim
from torch import nn
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms

from metrics import accuracy

def load_fashion_mnist_data(
    batch_size: int,
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[DataLoader, DataLoader]:
    # Transformations
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # Get Data
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data",
        train=True,
        transform=trans,
        download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data",
        train=False,
        transform=trans,
        download=True
    )
    train_dataset = DataLoader(
        mnist_train,
        batch_size,
        shuffle=True,
        num_workers=2
    )
    test_dataset = DataLoader(
        mnist_test,
        batch_size,
        shuffle=False,
        num_workers=2
    )
    return train_dataset, test_dataset

def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, std=0.01)

def create_net(n_inputs, n_outputs):
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_inputs, n_outputs),
        nn.LogSoftmax(dim=1),
    )
    net.apply(init_weights)
    return net

def create_deep_net(n_inputs, n_outputs):
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, n_outputs),
        nn.LogSoftmax(dim=1),
    )
    net.apply(init_weights)
    return net

def train(
    net: nn.Module,
    loss: _Loss,
    optimizer: Optimizer,
    train_dataset: DataLoader,
    eval_dataset: Optional[DataLoader] = None,
    n_epochs: int = 10,
):

    for epoch in range(n_epochs):
        loss_accum = 0.0
        accuracy_accum = 0.0
        samples_accum = 0.0
        net.train()
        for X, y in train_dataset:
            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_accum += l * y.numel()
            accuracy_accum += accuracy(y_hat, y)
            samples_accum += y.numel()
        print(
            f"Epoch {epoch}: "
            f"train loss = {(loss_accum / samples_accum):2.2e} "
            f"train accuracy = {(accuracy_accum / samples_accum):2.2f}"
        )

        if eval_dataset:
            loss_accum = 0.0
            accuracy_accum = 0.0
            samples_accum = 0.0
            net.eval()
            with torch.no_grad():
                for X, y in eval_dataset:
                    y_hat = net(X)
                    l = loss(y_hat, y)
                    loss_accum += l * y.numel()
                    samples_accum += y.numel()
                    accuracy_accum += accuracy(y_hat, y)
                print(
                    f"Epoch {epoch}: "
                    f"eval loss = {(loss_accum / samples_accum):2.2e} "
                    f"eval accuracy = {(accuracy_accum / samples_accum):2.2f}"
                )

def print_title(title: str):
    print("="*80)
    print(title.upper())
    print("="*80)

def main():
    N = 10_000
    BATCH_SIZE = 64
    train_dataset, eval_dataset = load_fashion_mnist_data(batch_size=BATCH_SIZE)
    loss = nn.NLLLoss()

    print_title("net 1")
    net1 = create_net(784, 10)
    optimizer = optim.SGD(net1.parameters(), lr=0.05)
    train(net1, loss, optimizer, train_dataset, eval_dataset, n_epochs=5)

    print_title("deep net")
    deep_net = create_deep_net(784, 10)
    optimizer = optim.SGD(deep_net.parameters(), lr=0.05)
    train(deep_net, loss, optimizer, train_dataset, eval_dataset, n_epochs=10)

if __name__ == "__main__":
    main()

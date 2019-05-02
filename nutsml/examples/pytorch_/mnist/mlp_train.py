"""
.. module:: cnn_train
   :synopsis: Example nuts-ml pipeline for training a MLP on MNIST
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import nutsflow as nf
import nutsml as nm
import numpy as np

from nutsml.network import PytorchNetwork
from utils import download_mnist, load_mnist


class Model(nn.Module):
    """MLP model"""

    def __init__(self, device):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

        # required properties of a model to be wrapped as PytorchNetwork!
        self.device = device  # 'cuda', 'cuda:0' or 'gpu'
        self.losses = nn.CrossEntropyLoss()  # can be list of loss functions
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def accuracy(y_true, y_pred):
    """Compute accuracy"""
    from sklearn.metrics import accuracy_score
    return 100 * accuracy_score(y_true, np.array(y_pred).argmax(1))


def evaluate(network, x, y):
    """Evaluate network performance (here accuracy)"""
    metrics = [accuracy]
    build_batch = (nm.BuildBatch(64)
                   .input(0, 'vector', 'float32')
                   .output(1, 'number', 'int64'))
    acc = zip(iter(x), iter(y)) >> build_batch >> network.evaluate(metrics)
    return acc


def predict(network, x, y):
    """Compute network outputs and print accuracy"""
    build_batch = nm.BuildBatch(64).input(0, 'vector', 'float32')
    preds = (zip(iter(x), iter(y)) >> build_batch >> network.predict() >> nf.Collect())
    acc = accuracy(y, preds)
    print('test acc', acc)


def train(network, epochs=3):
    """Train network for given number of epochs"""
    print('loading data...')
    filepath = download_mnist()
    x_train, y_train, x_test, y_test = load_mnist(filepath)

    build_batch = (nm.BuildBatch(64, verbose=False)
                   .input(0, 'vector', 'float32')
                   .output(1, 'number', 'int64'))

    for epoch in range(epochs):
        print('epoch', epoch + 1)
        losses = (zip(iter(x_train), iter(y_train)) >>
                  nf.PrintProgress(x_train) >> nf.Shuffle(100) >> build_batch >>
                  network.train() >> nf.Collect())
        acc_test = evaluate(network, x_test, y_test)
        acc_train = evaluate(network, x_train, y_train)
        print('train loss : {:.6f}'.format(np.mean(losses)))
        print('train acc  : {:.1f}'.format(acc_train))
        print('test acc   : {:.1f}'.format(acc_test))


if __name__ == '__main__':
    print('creating model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(device)
    network = PytorchNetwork(model)

    # network.load_weights()
    network.print_layers((28 * 28,))

    print('training network...')
    train(network, epochs=3)

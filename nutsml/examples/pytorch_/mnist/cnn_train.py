import requests
import pickle
import gzip

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import nutsflow as nf
import nutsml as nm
import numpy as np

from nutsml.network import PytorchNetwork
from pathlib import Path

BATCHSIZE = 64
EPOCHS = 3


def download_mnist():
    folder = Path("data/mnist")
    filename = "mnist.pkl.gz"
    fullpath = folder / filename
    url = "http://deeplearning.net/data/mnist/" + filename
    folder.mkdir(parents=True, exist_ok=True)
    if not fullpath.exists():
        content = requests.get(url).content
        fullpath.open("wb").write(content)
    return fullpath


def load_mnist(filepath):
    with gzip.open(filepath.as_posix(), "rb") as f:
        data = pickle.load(f, encoding="latin-1")
    (x_train, y_train), (x_valid, y_valid), _ = data
    return x_train, y_train, x_valid, y_valid


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
        )

        # required properties of a model to be wrapped as PytorchNetwork!
        self.device = device  # 'cuda', 'cuda:0' or 'gpu'
        self.losses = F.cross_entropy  # can be list of losses
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        """Forward pass through network for input x"""
        return self.layers(x)


build_batch = (nm.BuildBatch(BATCHSIZE, verbose=False)
               .input(0, 'image', 'float32', True)
               .output(1, 'number', 'int64'))
build_pred_batch = (nm.BuildBatch(BATCHSIZE, verbose=False)
                    .input(0, 'image', 'float32', True))
augment = nm.AugmentImage(0).by('identical', 0.5).by('rotate', 0.5, [-30, 30])
vec2img = nf.MapCol(0, lambda x: (x.reshape([28, 28]) * 255).astype('uint8'))
sample_gen = lambda x, y: zip(iter(x), iter(y))


def accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, np.array(y_pred).argmax(1))


def train(network, x, y):
    for epoch in range(EPOCHS):
        print('epoch', epoch + 1)
        losses = (sample_gen(x, y) >> nf.PrintProgress(x) >> vec2img >>
                  augment >> nf.Shuffle(1000) >> build_batch >>
                  network.train() >> nf.Collect())
        print('train loss:', np.mean(losses))


def validate(network, x, y):
    losses = (sample_gen(x, y) >> nf.PrintProgress(x) >> vec2img >>
              build_batch >> network.validate() >> nf.Collect())
    print('val loss:', np.mean(losses))


def predict(network, x, y):
    preds = (sample_gen(x, y) >> nf.PrintProgress(x) >> vec2img >>
             build_pred_batch >> network.predict() >> nf.Collect())
    acc = accuracy(y, preds)
    print('test acc', 100.0 * acc)


def evaluate(network, x, y):
    metrics = [accuracy]
    result = (sample_gen(x, y) >> nf.PrintProgress(x) >> vec2img >>
              build_batch >> network.evaluate(metrics))
    print(result)


def show_errors(network, x, y):
    make_label = nf.Map(lambda s: (s[0], 'true:%d  pred:%d' % (s[1], s[2])))
    filter_error = nf.Filter(lambda s: s[1] != s[2])
    view_image = nm.ViewImageAnnotation(0, 1, pause=1)

    preds = (sample_gen(x, y) >> vec2img >> build_pred_batch >>
             network.predict() >> nf.Map(np.argmax) >> nf.Collect())
    (zip(x, y, preds) >> vec2img >> filter_error >> make_label >>
     view_image >> nf.Consume())


def view_images(x, y):
    view_image = nm.ViewImageAnnotation(0, 1, pause=1)
    zip(x, y) >> vec2img >> augment >> view_image >> nf.Consume()


if __name__ == '__main__':
    print('loading data...')
    filepath = download_mnist()
    x_train, y_train, x_test, y_test = load_mnist(filepath)

    # print('viewing images...')
    # view_images(x_test, y_test)

    print('creating model...')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model(device)
    network = PytorchNetwork(model)
    # network.load_weights()
    network.print_layers((1, 28, 28))

    print('training ...')
    train(network, x_train, y_train)
    network.save_weights()

    print('validating ...')
    validate(network, x_test, y_test)

    print('predicting ...')
    predict(network, x_test, y_test)

    print('evaluating ...')
    evaluate(network, x_test, y_test)

    # print('showing errors ...')
    # show_errors(network, x_test, y_test)

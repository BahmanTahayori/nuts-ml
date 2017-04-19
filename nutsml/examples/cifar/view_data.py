"""
.. module:: view_data
   :synopsis: Example nuts-ml pipeline viewing CIFAR-10 image data
"""

from nutsflow import Take, Consume
from nutsml import ViewImage, PrintColType

if __name__ == "__main__":
    from cnn_train import load_samples

    train_samples, val_samples = load_samples()
    show_image = ViewImage(0, pause=1, figsize=(2, 2), interpolation='spline36')

    train_samples >> Take(10) >> PrintColType() >> show_image >> Consume()

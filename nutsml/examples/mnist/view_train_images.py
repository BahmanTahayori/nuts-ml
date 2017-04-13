"""
.. module:: view_train_images
   :synopsis: Example for showing images with annotation
"""

from nutsflow import Take, Consume
from nutsml import ViewImageAnnotation

if __name__ == "__main":
    from mlp_train import load_samples

    train, _ = load_samples()
    (train >> Take(10) >> ViewImageAnnotation(0, 1, pause=1) >> Consume())
